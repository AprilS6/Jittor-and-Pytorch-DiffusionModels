"""
训练脚本:
    训练新的模型，或加载预训练模型继续训练。选项可以指定部分参数，完整参数可以通过json文件指定。
命令行:
    python train.py [--option=<value>]
选项:
    --model_path: 预训练模型路径
    --config_path: 配置文件路径
    --data_root: 数据集根目录
    --dataset_name: 数据集名称
    --num_epochs: 训练轮数
    --batch_size: 批大小
    --num_workers: 线程数
    --T: 时间步数
    --lr: 学习率
    --device: 设备类型
    --log: 是否记录日志
    --log_root: 日志根目录
    --checkpoint: 检查点间隔，epoch到检查点后备份一次模型
    --only_checkpoint_max: 只保留最新的检查点
"""

from model import DDPM, default_config
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--T', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--device', type=str)
    parser.add_argument('--log', type=bool)
    parser.add_argument('--log_root', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--only_checkpoint_max', type=bool)
    args = parser.parse_args()
    config = default_config.copy() if args.config_path is None else json.load(open(args.config_path))
    config.update([(k, v) for k, v in vars(args).items() if v is not None])
    model = DDPM(config, args.model_path)
    model.train()
    
if __name__ == '__main__':
    main()