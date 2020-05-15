import torch
from torch.utils.data import DataLoader
import numpy as np
import logging

from fer2013_dataset import FER2013Dataset


def main(args):
    train_dataset = FER2013Dataset(args.data_path, "Training")
    public_test_dataset = FER2013Dataset(args.data_path, "PublicTest")
    private_test_dataset = FER2013Dataset(args.data_path, "PrivateTest")


    # Get class weights from class occurences in the dataset. 
    dataset_summary = train_dataset.get_summary_statistics()
    class_weights = (1/dataset_summary["class_occurences"])
    class_weights = class_weights / np.sum(class_weights)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    public_test_loader = DataLoader(public_test_dataset, batch_size=32, shuffle=False, pin_memory=True)
    private_test_loader = DataLoader(private_test_dataset, batch_size=32, shuffle=False, pin_memory=True)
    
    for idx, batch in enumerate(train_loader):
        print(batch['image'].shape)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to the full dataset")

    args = parser.parse_args()

    main(args)