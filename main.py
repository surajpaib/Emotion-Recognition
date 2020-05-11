import torch
import logging
from fer2013_dataset import FER2013Dataset
from torch.utils.data import DataLoader

def main(args):
    train_dataset = FER2013Dataset(args.train_data)
    test_dataset = FER2013Dataset(args.test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, pin_memory=True)
    
    for idx, batch in enumerate(train_loader):
        print(batch['image'].shape)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_data", help="Path to train data")
    parser.add_argument("test_data", help="Path to test data")

    args = parser.parse_args()

    main(args)