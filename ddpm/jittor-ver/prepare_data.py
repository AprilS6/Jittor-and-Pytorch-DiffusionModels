import jittor as jt
from typing import Callable

def load_data(
        root: str="../../data",
        dataset_name: str = "cifar10",
        train: bool = True,
        download: bool = True,
        transform: Callable | None = None
    ):
    """
    下载CIFAR-10数据集，并返回标准化的Dataset
    Args:
        root: 数据集根目录
        dataset_name: 数据集名称
        download: 是否自动下载数据集
        train: 是否为训练集
        transform: 自定义数据预处理流程（若为None则使用内置默认预处理流程）
    Return:
        Dataset
    """
    if dataset_name == "cifar10":
        if transform is None:
            transform = jt.transform.Compose([
                jt.transform.ToTensor(), # [0, 255] -> [0, 1]
                jt.transform.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0, 1] -> [-1,1]
                jt.transform.RandomHorizontalFlip() # 随机水平翻转
            ])
        dataset = jt.dataset.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
    else:
        raise ValueError("Unsupported dataset name: {}".format(dataset_name))
    return dataset