import jittor as jt
from load_data import get_dataset
from jittor.dataset import DataLoader


def main():
    jt.flags.use_cuda = 1
    dataset = get_dataset(dataset_name="cifar10")
    dataloader = DataLoader(dataset, batch_size=64, num_workers=2, shuffle=True)
    for i, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        if i > 5:
            break


if __name__ == '__main__':
    main()