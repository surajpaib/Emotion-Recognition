import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging

from utils.metrics import Metrics
from utils.utils import convertModel
from models.model import Model
from fer2013_dataset import FER2013Dataset

# Reproducibility Settings
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test(args):
    # Get hardware device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    # Load dataset based on usage specified in "test_split" argument, either PrivateTest or PublicTest
    test_dataset = FER2013Dataset(args.data_path, args.test_split)

    # Load dataloader with a large size and shuffle false since we dont care about the order while testing
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True)


    # Load model definition from the model config, do not initialize weights for test since they will be loaded
    model = Model(args.model_config, initialize_weights=False)

    # Load model state and weights from the checkpoint. convertModel converts any DataParallel model to current device.
    model = convertModel(args.load_model, model).to(device)

    # Model in evaluation mode
    model.eval()

    # Intialize metric logger object
    metrics = Metrics()

    # Set no grad to disable gradient saving. 
    with torch.no_grad():

        # Iterate over each batch in the test loader
        for idx, batch in enumerate(test_loader):

            # Move the batch to the device, needed explicitly if GPU is present
            image, target = batch["image"].to(device), batch["emotion"].to(device)

            # Forward pass
            out = model(image)

            # Metrics and sample predictions
            metrics.update_test({"predicted": out, "ground_truth": target})


    # Get confusion matrix plot
    metrics.confusion_matrix_plot(test_dataset.get_class_mapping(), args.test_split)

    # Save other statistics to the csv report
    test_report = metrics.get_report(mode="test")
    test_report.to_csv("results/{}_testreport.csv".format(args.load_model.format("/")[-1].split(".")[0]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the full dataset", default="data/fer2013/fer2013/fer2013.csv")
    
    # Model configuration for the experiment
    parser.add_argument("--model_config", help="Path to the model configuration json", default="results/bestModel.json")    
    parser.add_argument("--load_model", help="Path to the model tar file to load", default="results/bestModel.pth.tar")    
    
    # Test data split to run the tests on
    parser.add_argument("--test_split", help="Label of the test split to evaluate on", default="PrivateTest")    

    args = parser.parse_args()

    test(args)