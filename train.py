import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import logging
from tqdm import tqdm

from utils.metrics import Metrics
from utils.utils import get_loss, save_checkpoint, apply_transforms
from utils.viz import visualize_filters

from models.model import Model
from fer2013_dataset import FER2013Dataset

# Reproducibility Settings
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(args):
    # Get hardware device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check if weights and biases integration is enabled. 
    if args.wandb == 1:
        import wandb
        wandb.init(entity='surajpai', project='FacialEmotionRecognition',config=vars(args))

    # Get the dataset with "Training" usage.
    dataset = FER2013Dataset(args.data_path, "Training")

    # Randomly split the dataset into train and validation based on the specified train_split argument
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*args.train_split), len(dataset) - int(len(dataset)*args.train_split)])


    logging.info('Samples in the training set: {}\n Samples in the validation set: {} \n\n'.format(len(train_dataset), len(validation_dataset)))

    # Get class weights as inverse of frequencies from class occurences in the dataset.
    dataset_summary = dataset.get_summary_statistics()
    class_weights = (1/dataset_summary["class_occurences"])
    class_weights = torch.Tensor(class_weights / np.sum(class_weights)).to(device)

    # Train loader and validation loader initialized with batch_size as specified and randomly shuffled
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)


    # Model initialization
    model = torch.nn.DataParallel(Model(args.model_config)).to(device)

    # Set torch optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
    )

    # Get loss for training the network from the utils get_loss function
    criterion = get_loss(args, class_weights)
    bestLoss = -1000

    # Create metric logger object
    metrics = Metrics(upload=args.wandb)

    # Define augmentation transforms, if --augment is enabled
    if args.augment == 1:
        transform = transforms.RandomChoice([transforms.RandomHorizontalFlip(p=0.75), transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(1.2, 1.2), shear=15),
                                            transforms.ColorJitter()])


    # Start iterating over the total number of epochs set by epochs argument
    for n_epoch in range(args.epochs):

        # Reset running metrics at the beginning of each epoch.
        metrics.reset()

        # Utils logger
        logging.info(' Starting Epoch: {}/{} \n'.format(n_epoch, args.epochs))

        '''

        TRAINING

        '''

        # Model in train mode for batch-norm and dropout related ops.
        model.train()

        # Iterate over each batch in the train loader
        for idx, batch in enumerate(tqdm(train_loader)):

            # Reset gradients
            optimizer.zero_grad()

            # Apply augmentation transforms, if --augment is enabled
            if args.augment == 1 and n_epoch % 2 == 0:
                batch = apply_transforms(batch, transform)

            # Move the batch to the device, needed explicitly if GPU is present
            image, target = batch["image"].to(device), batch["emotion"].to(device)

            # Run a forward pass over images from the batch
            out = model(image)

            # Calculate loss based on the criterion set
            loss = criterion(out, target)

            # Backward pass from the final loss
            loss.backward()

            # Update the optimizer
            optimizer.step()

            # Update metrics for this batch
            metrics.update_train({"loss": loss.item(), "predicted": out, "ground_truth": target})
        '''

        VALIDATION

        '''
        
        logging.info(' Validating on the validation split ... \n \n')
        
        # Model in eval mode.
        model.eval()
        
        # Set no grad to disable gradient saving. 
        with torch.no_grad():

            # Iterate over each batch in the val loader
            for idx, batch in enumerate(val_loader):

                # Move the batch to the device, needed explicitly if GPU is present
                image, target = batch["image"].to(device), batch["emotion"].to(device)

                # Forward pass
                out = model(image)

                # Calculate loss based on the criterion set
                loss = criterion(out, target)

                # Metrics and sample predictions updated for validation batch
                metrics.update_val({"loss": loss.item(), "predicted": out, "ground_truth": target, "image": image, "class_mapping": dataset.get_class_mapping()})

        # Display metrics at the end of each epoch
        metrics.display()

        # Weight Checkpointing to save the best model on validation loss
        save_path = "./saved_models/{}.pth.tar".format(args.model_config.split('/')[-1].split('.')[0])
        bestLoss = min(bestLoss, metrics.metric_dict["loss@val"])
        is_best = (bestLoss == metrics.metric_dict["loss@val"])
        save_checkpoint({
                    'epoch': n_epoch,
                    'state_dict': model.state_dict(),
                    'bestLoss': bestLoss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, save_path)


    # After training is completed, if weights and biases is enabled, visualize filters and upload final model.
    if args.wandb == 1:
        visualize_filters(model.modules())
        wandb.save(save_path)


    # Get report from the metrics logger
    train_report, val_report = metrics.get_report()

    # Save the report to csv files
    train_report.to_csv("{}_trainreport.csv".format(save_path.rstrip(".pth.tar")))
    val_report.to_csv("{}_valreport.csv".format(save_path.rstrip(".pth.tar")))
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Data related
    parser.add_argument("--data_path", help="Path to the full dataset", type=str, default="data/fer2013/fer2013/fer2013.csv")
    parser.add_argument("--augment", help="Enable data augmentation", type=int, default=1)

    # Model configuration for the experiment
    parser.add_argument("--model_config", help="Path to the model configuration json", type=str, default="config/Baseline.json")

    # Training hyperparameters
    parser.add_argument("--epochs", help="Number of epochs to train",type=int, default=100)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--train_split", help="Train-valid split", type=float, default=0.8)

    # Loss-specific hyperparameters
    parser.add_argument("--balanced_loss", help="if True, weights losses according to class instances", type=int, default=0)
    parser.add_argument("--loss", help="Type of loss to be used", type=str, default='CrossEntropyLoss')

    parser.add_argument("--wandb", help="Wandb integration", type=int, default=0)

    args = parser.parse_args()

    train(args)