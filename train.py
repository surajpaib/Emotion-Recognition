import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import trange

from utils.utils import get_batch_evaluation_metrics, get_image_predictions, get_loss, concatenate_metrics, wandb_log, save_checkpoint
from models.model import Model
from fer2013_dataset import FER2013Dataset


def main(args):

    if args.wandb:
        import wandb
        wandb.init(entity='surajpai', project='FacialEmotionRecognition')

    dataset = FER2013Dataset(args.data_path, "Training")
    public_test_dataset = FER2013Dataset(args.data_path, "PublicTest")
    private_test_dataset = FER2013Dataset(args.data_path, "PrivateTest")


    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*args.train_split), len(dataset) - int(len(dataset)*args.train_split)])


    logging.info('Samples in the training set: {}\n Samples in the validation set: {} \n\n'.format(len(train_dataset), len(validation_dataset)))

    # Get class weights from class occurences in the dataset. 
    dataset_summary = dataset.get_summary_statistics()
    class_weights = (1/dataset_summary["class_occurences"])
    class_weights = class_weights / np.sum(class_weights)

    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    

    # Model initialization
    model = Model(args.model_config)

    # Set torch optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.5e-4,
    )

    # Get loss for training the network
    criterion = get_loss(args, class_weights)
    bestLoss = -1000    
    
    for n_epoch in range(args.epochs):
        # Utils logger
        logging.info(' Starting Epoch: {}/{} \n'.format(n_epoch, args.epochs))
        t = trange(len(train_loader), desc='Loss', leave=True)


        '''
        
        TRAINING 
        
        '''

        model.train()
        train_eval_metrics = {"loss": 0.0, "accuracy":0.0}

        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            image, target = batch["image"], batch["emotion"]

            out = model(image)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()
        
            t.set_description("Loss : {}".format(loss.item()))
            t.update()

            train_eval_metrics = get_batch_evaluation_metrics(train_eval_metrics, out, target, loss)


        for key in train_eval_metrics:
            train_eval_metrics[key] = train_eval_metrics[key]/len(train_loader)
        


        logging.info(" Train metrics at end of epoch {}: {} \n".format(n_epoch, train_eval_metrics))


        '''
        
        VALIDATION 
        
        '''
        logging.info(' Validating on the validation split ... \n \n')

        val_eval_metrics = {"loss": 0.0, "accuracy":0.0}
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                image, target = batch["image"], batch["emotion"]
                out = model(image)

                loss = criterion(out, target)

                # Metrics and sample predictions
                val_eval_metrics = get_batch_evaluation_metrics(val_eval_metrics, out, target, loss)

                if idx == 0:
                    image_predictions = get_image_predictions(image, target, out, dataset.get_class_mapping())

            for key in train_eval_metrics:
                val_eval_metrics[key] = val_eval_metrics[key]/len(val_loader)
            logging.info(" Val metrics at end of epoch {}: {} \n\n\n".format(n_epoch, val_eval_metrics))

            

        # Weight Checkpointing to save the best model on validation loss
        bestLoss = min(bestLoss, val_eval_metrics["loss"]) 
        is_best = (bestLoss == val_eval_metrics["loss"])
        save_checkpoint({
                    'epoch': n_epoch,
                    'state_dict': model.state_dict(),
                    'bestLoss': bestLoss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)



        # Aggregate metrics to send to weights and biases
        metrics = concatenate_metrics(train_eval_metrics, val_eval_metrics)

        if args.wandb:
            wandb_log(image_predictions, metrics)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the full dataset", default="data/fer2013/fer2013/fer2013.csv")
    
    # Model configuration for the experiment
    parser.add_argument("--model_config", help="Path to the model configuration json", default="config/Baseline.json")

    # Training hyperparameters
    parser.add_argument("--epochs", help="Number of epochs to train", default=100)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=128)
    parser.add_argument("--train_split", help="Train-valid split", type=float, default=0.8)
    
    # Loss-specific hyperparameters
    parser.add_argument("--balanced_loss", help="if True, weights losses according to class instances", type=bool, default=False)
    parser.add_argument("--loss", help="Type of loss to be used", type=str, default='CrossEntropyLoss')
    
    parser.add_argument("--wandb", help="Wandb integration", type=bool, default=False)



    args = parser.parse_args()

    main(args)