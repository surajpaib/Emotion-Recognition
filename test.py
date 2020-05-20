import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging

from utils.metrics import Metrics
from utils.utils import convertModel
from models.model import Model
from fer2013_dataset import FER2013Dataset


def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = FER2013Dataset(args.data_path, args.test_split)

    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=True, pin_memory=True)

    model = Model(args.model_config)
    model = convertModel(args.load_model, model).to(device)


    model.eval()

    metrics = Metrics()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image, target = batch["image"].to(device), batch["emotion"].to(device)
            out = model(image)

            # Metrics and sample predictions
            metrics.update_test({"predicted": out, "ground_truth": target})



    test_report = metrics.get_report(mode="test")
    test_report.to_csv("results/{}_testreport.csv".format(args.load_model))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the full dataset", default="data/fer2013/fer2013/fer2013.csv")
    
    # Model configuration for the experiment
    parser.add_argument("--model_config", help="Path to the model configuration json", default="config/Baseline.json")    
    parser.add_argument("--load_model", help="Path to the model tar file to load", default="checkpoint.pth.tar")    
    parser.add_argument("--test_split", help="Label of the test split to evaluate on", default="PrivateTest")    

    args = parser.parse_args()

    test(args)