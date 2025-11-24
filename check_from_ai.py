import torch
import os
from cdm.model import create_model

# ================= 配置 =================
# 确保路径正确
base_path = './models/base.ckpt'
task_path = './incre_val/mvtec_setting1/task1_best.ckpt' # 替换为您实际的任务路径
config_path = 'models/cdad_mvtec.yaml'
# =======================================

def check_weight_match(model_weight, file_weight, name):
    if torch.equal(model_weight.cpu(), file_weight.cpu()):
        print(f"✅ {name}: 权重匹配！(加载成功)")
        return True
    else:
        diff = (model_weight.cpu() - file_weight.cpu()).abs().sum().item()
        print(f"❌ {name}: 权重不匹配！(加载失败)")
        print(f"   差异总量: {diff}")
        print(f"   -> 说明模型还在使用随机初始化的参数，或者旧参数。")
        return False

def main():
    print(f"--- 开始验证权重加载状态 ---")
    
    # 1. 创建模型 (此时所有参数都是随机的)
    print("1. 创建模型...")
    model = create_model(config_path).cpu()
    
    # 2. 模拟您的操作：先加载 Base
    print(f"\n2. 加载 Base: {base_path}")
    if os.path.exists(base_path):
        base_sd = torch.load(base_path, map_location='cpu')
        # 过滤掉 AMN
        select_weights = {k: v for k, v in base_sd.items() if 'control_model' not in k}
        model.load_state_dict(select_weights, strict=False)
    else:
        print("   (Base文件不存在，跳过)")

    # 3. 模拟您的操作：加载 Task (但不拆包，模拟错误写法)
    print(f"\n3. 加载 Task (模拟错误写法，不拆包): {task_path}")
    # === 错误写法 ===
    # 这就是导致 reconstruction 好了但 sample 坏了的原因
    task_wrapper = torch.load(task_path, map_location='cpu') 
    model.load_state_dict(task_wrapper, strict=False) 
    # ================

    # 4. 开始对比真值
    print(f"\n4. 揭晓真相：对比内存中的权重 vs 文件里的权重")
    
    # 提取文件里的真实权重 (拆包)
    if 'state_dict' in task_wrapper:
        real_task_weights = task_wrapper['state_dict']
    else:
        real_task_weights = task_wrapper

    # --- 检查 AMN (控制网络) ---
    # 既然 sample 是坏的，AMN 肯定没加载进去
    print("\n[检查 AMN (Anomaly-Masked Network)]")
    amn_key = 'control_model.input_hint_block.0.weight' # AMN 第一层
    if amn_key in real_task_weights:
        check_weight_match(model.state_dict()[amn_key], real_task_weights[amn_key], "AMN 第一层卷积")
    else:
        print(f"   无法在文件中找到 {amn_key}")

    # --- 检查 U-Net (微调部分) ---
    # 如果 U-Net 也没加载进去，那就是用的 Base 里的旧权重
    print("\n[检查 U-Net (Diffusion Model)]")
    unet_key = 'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight' # 某个深层权重
    if unet_key in real_task_weights:
        check_weight_match(model.state_dict()[unet_key], real_task_weights[unet_key], "U-Net 某层权重")
    
    print("\n------------------------------------------------")
    print("结论预测：")
    print("如果上面显示 ❌，说明您虽然修好了 VAE，但 Task 权重依然没加载进去。")
    print("AMN 是随机的 -> Control 信号是乱的 -> Sample 图是乱的。")

if __name__ == "__main__":
    main()