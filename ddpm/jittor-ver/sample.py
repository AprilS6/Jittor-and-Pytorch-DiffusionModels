"""
采样脚本：
    随机采样噪声，生成图像。一般通过指定model_path和config_path来使用已训练好的模型进行采样。
命令行:
    python samplt.py [--option=<value>]
选项：
    --model_path: 模型路径
    --config_path: 配置文件路径
    --num_samples: 采样数量
    --show_denoised: 展示降噪过程，默认展示16张
    --img_merge: 图片拼接方式，默认不拼接
    --gif_time: gif总时间
    --gif_dt: gif帧间隔
    --sample_root: 采样结果保存路径，默认在samples下
    --batch_size: 批大小
    --num_workers: 线程数
    --device: 设备
    --log: 是否记录日志
    --log_root: 日志保存路径, 默认在logs下
"""

from model import DDPM, default_config
import argparse
import json
import numpy as np
from PIL import Image
import os
import time

def gif_dt(n: int, d: int, total_sum: int):
    a0 = (2 * total_sum / n - (n - 1) * d) / 2
    assert a0 > 0
    return [int(a0 + d * i) for i in range(n)]

def merge_xs(tbx):
    tbi = []
    for bx in tbx:
        bx = bx.numpy().astype(np.uint8)
        b, h, w, c = bx.shape
        cols = int(b ** 0.5)
        rows = (b + cols - 1) // cols
        total = rows * cols
        padded = np.full((total, h, w, c), 255, dtype=np.uint8)
        padded[:b] = bx
        padded = padded.reshape(rows, cols, h, w, c)
        grid = []
        for row in range(rows):
            row_combined = np.concatenate(padded[row], axis=1)
            grid.append(row_combined)
        full_image = np.concatenate(grid, axis=0)
        full_image = Image.fromarray(full_image)
        tbi.append(full_image)
    return tbi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_root', type=str, default=os.path.join('samples', time.strftime("%Y%m%d-%H%M%S")))
    
    # parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str)
    parser.add_argument('--show_denoised', type=int, default=16)
    parser.add_argument('--gif_time', type=int, default=8000)
    parser.add_argument('--gif_dt', type=int, default=None)
    
    parser.add_argument('--model_path', type=str, default="eps_best.pkl")
    parser.add_argument('--config_path', type=str, default="eps_best.json")
    parser.add_argument('--log_root', type=str, default='logs/temp')
    parser.add_argument('--log', type=bool, default=True)
    args = parser.parse_args()
    
    config = default_config.copy() if args.config_path is None else json.load(open(args.config_path))
    config.update([(k, v) for k, v in vars(args).items() if v is not None])
    model = DDPM(config, args.model_path)
    xs = model.rand_sample_x0(
        batch_size=args.batch_size,
        show_denoised=args.show_denoised
    )
    
    if not os.path.exists(args.sample_root):
        os.makedirs(args.sample_root)
    imgs = merge_xs(xs)
    args.gif_dt = args.gif_time // (len(imgs) * len(imgs)) if args.gif_dt is None else args.gif_dtr
    if len(imgs) > 1 and args.gif_time > 0:
        imgs[0].save(
            os.path.join(args.sample_root, "0.gif"),
            save_all=True,
            append_images=imgs[1:],
            duration=gif_dt(len(imgs), args.gif_dt, args.gif_time),
            loop=0
        )
    for i in range(len(imgs)):
        imgs[i].save(os.path.join(args.sample_root, f'{i}.png'))
    print(f"Saved to {args.sample_root}")
    
if __name__ == '__main__':
    main()