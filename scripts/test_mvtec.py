import sys
import os
sys.path.append(os.getcwd())
from share import *
import torch
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
from cdm.model import create_model
from utils.eval_helper import dump, save_metrics, merge_together, performances
from scipy.ndimage import gaussian_filter
import cv2
from utils.util import cal_anomaly_map, log_local, setup_seed
from tqdm import tqdm
from data.mvtecad_dataloader import MVTecDataset_cad

def main(args):

    resume_path = f'./incre_val/mvtec_setting{args.setting}/task{args.task}_best.ckpt'

    setup_seed(args.seed)

    model = create_model('models/cdad_mvtec.yaml').cpu()

    log = f'demo_task{args.task}'

# ✅ 修正后的代码
    print(f"Loading weights from {resume_path}")
    checkpoint = torch.load(resume_path, map_location='cpu')

    # 【核心修复】检查是否包含 'state_dict' 键（拆包裹）
    if 'state_dict' in checkpoint:
        weights = checkpoint['state_dict']
    else:
        weights = checkpoint

    # 加载参数到模型
    # strict=False 是必须的，以防有一些无关的键不匹配
    msg = model.load_state_dict(weights, strict=False)
    print(f"权重加载完成。缺失键(Missing): {len(msg.missing_keys)} 个, 多余键(Unexpected): {len(msg.unexpected_keys)} 个")

    # Misc
    dataset, _ = MVTecDataset_cad('test', args.data_path, args.setting)
    dataloader = DataLoader(dataset[args.task], num_workers=8, batch_size=args.batch_size, shuffle=False)

    model = model.cuda()

    model.eval()

    result = {'clsname':[], 'filename':[], 'pred':[], 'mask':[], 'input':[]}

    evl_dir = "TEST/mvtec"
    os.makedirs(f"{evl_dir}/image", exist_ok=True)
    os.makedirs(f"{evl_dir}/log/setting{args.setting}/", exist_ok=True)

    with torch.no_grad():
        for input in tqdm(dataloader):
            output = model.log_images_test(input)
            model.pretrained_resnet50.eval()

            input_img = output['reconstruction']
            input_features = model.pretrained_resnet50(input_img)

            images = output
            images['hint'] = input['hint']
            log_local(images, input["filename"], f'{evl_dir}/image')
            output_img = images['samples']
            output_features = model.pretrained_resnet50(output_img.cuda())

            input_features = [input_features[i] for i in model.layers_]
            output_features = [output_features[i] for i in model.layers_]

            anomaly_map, _ = cal_anomaly_map(input_features, output_features, input_img.shape[-1], amap_mode='a', dis=model.distance)

            for i in range(anomaly_map.shape[0]):
                anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=5)

            anomaly_map = torch.from_numpy(anomaly_map)
            anomaly_map_prediction = anomaly_map.unsqueeze(dim=1)

            result['filename'] += input["filename"]           # list:12
            result['pred'] += list(anomaly_map_prediction.cpu().numpy())     # B x 1 x H x W
            result['mask'] += list(input["mask"].cpu().numpy())     # B x 1 x H x W
            result['clsname'] += input["clsname"]
            result['input'] += list(input['jpg'].numpy())

        root = os.path.join(f'{evl_dir}/image_{log}/')

        pred_feature = result['pred']

        for i in range(len(pred_feature)):

            name = result["filename"][i][-7:-4]

            if not os.path.exists(os.path.join(root, result["filename"][i][:-7])):
                os.makedirs(os.path.join(root, result["filename"][i][:-7]))

            anomaly_map = torch.from_numpy(pred_feature[i][0, :, :])

            #Heatmap
            anomaly_map_new = np.round(255.0 * ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())))

            anomaly_map_new = anomaly_map_new.cpu().numpy().astype(np.uint8)

            heatmap = cv2.applyColorMap(anomaly_map_new, colormap=cv2.COLORMAP_JET)

            image = (torch.from_numpy(result['input'][i]) * 0.5 + 0.5) * 255.0
            image = image.permute(1, 2, 0).numpy().astype('uint8')

            out_heat_map = cv2.addWeighted(heatmap.copy(), 0.5, image.copy(), 0.5, 0, image.copy())

            heatmap_name = "{}-heatmap.png".format(name)
            cv2.imwrite(root + result["filename"][i][:-7] + heatmap_name, out_heat_map)


    print("Gathering final results ...")

    # evl_metrics = {'auc': [{'name': 'max'}, {'name': 'pixel'}, {'name': 'pro'}, {'name': 'apsp'}]}
    evl_metrics = {'auc': [{'name': 'apsp'}, {'name': 'pro'}]}
    
    fileinfos, preds, masks = merge_together(result)

    ret_metrics = performances(fileinfos, preds, masks, evl_metrics)

    path = f"{evl_dir}/log/setting{args.setting}/task{args.task}.csv"

    save_metrics(ret_metrics, evl_metrics, path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CDAD")

    parser.add_argument("--data_path", default="./data/mvtec_anomaly_detection", type=str)

    parser.add_argument("--setting", default=1, type=int)

    parser.add_argument("--task", default=0, type=int)

    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--batch_size", default=12, type=int)

    args = parser.parse_args()

    main(args)



