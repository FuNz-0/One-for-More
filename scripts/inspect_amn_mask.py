import sys
import os
import torch
import numpy as np

# 将项目根目录加入 python path
sys.path.append(os.getcwd())

from cdm.model import create_model
from utils.util import setup_seed

# ================= 配置 =================
LOG_FILE = "amn_details.txt"
BATCH_SIZE = 1
# =======================================

def write_to_file(text):
    """辅助函数：追加写入文件，同时在控制台打印进度"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_tensor_slice(name, tensor):
    """
    【核心修改】查看 4D 张量的具体长相
    策略：只取 Batch=0, Channel=0 的中心 10x10 区域打印
    """
    # 1. 基础信息
    shape = list(tensor.shape)
    write_to_file(f"[{name}]")
    write_to_file(f"  整体维度: {shape} (Batch, Channel, Height, Width)")
    write_to_file(f"  数值统计: Mean={tensor.mean():.4f}, Std={tensor.std():.4f}, Min={tensor.min():.4f}, Max={tensor.max():.4f}")
    
    # 2. 提取切片 (Batch 0, Channel 0)
    # 如果维度不够4维（比如Mask），做特殊处理
    if len(shape) == 4:
        slice_2d = tensor[0, 0, :, :]
    elif len(shape) == 3:
        slice_2d = tensor[0, :, :]
    elif len(shape) == 2:
        slice_2d = tensor
    else:
        write_to_file("  (维度太少，不打印切片)")
        return

    h, w = slice_2d.shape
    
    # 3. 截取中心 10x10 区域
    center_h, center_w = h // 2, w // 2
    window = 5
    start_h = max(0, center_h - window)
    end_h = min(h, center_h + window)
    start_w = max(0, center_w - window)
    end_w = min(w, center_w + window)
    
    write_to_file(f"  --- 真实数值采样 (第0通道, 中心 {end_h-start_h}x{end_w-start_w} 区域) ---")
    
    for r in range(start_h, end_h):
        row_vals = []
        for c in range(start_w, end_w):
            val = slice_2d[r, c].item()
            # 格式化一下，保留3位小数，让排版整齐点
            row_vals.append(f"{val:7.3f}")
        write_to_file(f"  Row {r:03d}: " + " ".join(row_vals))
    write_to_file("-" * 50)

def visualize_mask_to_file(mask, h, w):
    """
    Mask 可视化：使用字符画展示屏蔽逻辑
    """
    write_to_file("\n" + "="*20 + " [Mask 矩阵深度解剖] " + "="*20)
    write_to_file(f"Mask 物理形状: {list(mask.shape)}")
    
    # 选取中心点作为观察对象 (Query)
    center_r, center_c = h // 2, w // 2
    center_idx = center_r * w + center_c
    
    write_to_file(f"观察中心点坐标: ({center_r}, {center_c}) [Token Index: {center_idx}]")
    write_to_file(f"说明: 下图展示了当 Transformer 处理这个中心点时，它能看到周围哪些点。")
    write_to_file("图例: [      ] = 0.0 (可见/远处)   [  XX  ] = -inf (屏蔽/邻居)")
    
    # 取出这一行：代表“中心点”对所有其他点的注意力 Mask
    view_map = mask[center_idx].reshape(h, w)
    
    # 打印中心点周围区域 (范围稍大一点，看清楚边界)
    window = 6
    r_start = max(0, center_r - window)
    r_end = min(h, center_r + window + 1)
    c_start = max(0, center_c - window)
    c_end = min(w, center_c + window + 1)
    
    write_to_file(f"\n--- Mask 视野图 (中心 {r_end-r_start}x{c_end-c_start} 区域) ---")
    
    for r in range(r_start, r_end):
        row_str = f"Row {r:02d}: "
        for c in range(c_start, c_end):
            val = view_map[r, c].item()
            if val == float('-inf'):
                if r == center_r and c == center_c:
                    row_str += "[  自  ]" # 自己
                else:
                    row_str += "[  XX  ]" # 邻居 (屏蔽)
            else:
                row_str += "[      ]" # 远处 (可见)
        write_to_file(row_str)
    
    write_to_file("="*60 + "\n")

def main():
    setup_seed(42)
    
    # 1. 清空日志
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== AMN 模块数据流透视 ===\n\n")
    
    print(f"1. 初始化模型 (使用 CUDA)...")
    # 必须用 cuda，否则 AMN 内部的 .cuda() 会报错
    model = create_model('models/cdad_mvtec.yaml').cuda()
    amn = model.control_model
    
    # 简单加载 Base 权重，保证卷积层有参数，输出数值不是乱码
    base_path = './models/base.ckpt'
    if os.path.exists(base_path):
        print("   加载 Base 权重...")
        base_sd = torch.load(base_path, map_location='cpu')
        select_base = {k: v for k, v in base_sd.items() if 'control_model' not in k}
        model.load_state_dict(select_base, strict=False)
    
    print(f"2. 安装探针...")

    # --- 探针 A: Monkey Patch (拦截 Mask) ---
    original_generate_mask = amn.generate_mask
    def verbose_generate_mask(feature_size, neighbor_size):
        mask = original_generate_mask(feature_size, neighbor_size)
        write_to_file(f">>> [探针 A] Mask 生成完毕")
        visualize_mask_to_file(mask, feature_size[0], feature_size[1])
        return mask
    amn.generate_mask = verbose_generate_mask

    # --- 探针 B: Forward Hook (拦截数据数值) ---
    def io_hook(module, input, output):
        write_to_file(f">>> [探针 B] I/O 数据捕获")
        
        # Input
        if input and len(input) > 0:
            log_tensor_slice("输入 (Hint Image)", input[0])
        else:
            write_to_file("  [输入] 未捕获到位置参数")

        # Output
        log_tensor_slice("输出 (Control Feature)", output)
        write_to_file("\n" + "="*60 + "\n")

    amn.register_forward_hook(io_hook)

    # 3. 构造假数据并运行
    print(f"3. 运行推理...")
    # 构造符合 MVTec 规格的数据 (Batch, 3, 256, 256)，并在 GPU 上
    dummy_input = torch.randn(BATCH_SIZE, 3, 256, 256).cuda()
    
    with torch.no_grad():
        _ = amn(dummy_input)

    print(f"\n>>> ✅ 分析完成！")
    print(f">>> 请打开 [{LOG_FILE}] 查看：")
    print("    1. Mask 的字符画结构")
    print("    2. 输入输出张量的真实数值采样")

if __name__ == "__main__":
    main()