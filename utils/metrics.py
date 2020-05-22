import torch
import wandb
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Metrics:
    def __init__(self, upload=0):
        """
        Metrics class to store the results during training/validation/testing

        Args:
        upload: If 1, upload to weights and biases
        """
        self.upload = bool(upload)

        # Initialize all data structures for the metrics class
        self.reset()

    def reset(self):
        """
        Intialize metric dict, lists for losses, predictions and true values.
        Lists of images is also initialized but used only if upload is set to 1
        """
        self.metric_dict = {}

        self.train_losses = []
        self.val_losses = []

        self.train_predicted = []
        self.train_ground_truth = []

        self.val_predicted = []
        self.val_ground_truth = []

        self.test_predicted = []
        self.test_ground_truth = []

        self.predicted_images = []


    def update_train(self, metric_dict):
        """
        Update predicted, true values and losses for training 
        """
        self.train_losses.append(metric_dict["loss"])

        for pred in metric_dict["predicted"]:
            self.train_predicted.append(pred.tolist())

        for target in metric_dict["ground_truth"]:
            self.train_ground_truth.append(target.tolist())


    def update_val(self, metric_dict):
        """
        Update predicted, true values and losses for validation
        """

        self.class_mapping = metric_dict["class_mapping"]
        pred_classes = metric_dict["predicted"].max(dim=1)[1]
        self.val_losses.append(metric_dict["loss"])

        for pred in metric_dict["predicted"]:
            self.val_predicted.append(pred.tolist())

        for target in metric_dict["ground_truth"]:
            self.val_ground_truth.append(target.tolist())

        # Store images with their predictions
        for idx in range(metric_dict["image"].shape[0]):
            self.predicted_images.append({"Image": metric_dict["image"][idx], \
                 "Predicted": self.class_mapping[pred_classes[idx]],\
                      "Ground Truth": self.class_mapping[metric_dict["ground_truth"][idx]]})


    def update_test(self, metric_dict):
        """
        Update predicted, true values and losses for testing

        """
        pred_classes = metric_dict["predicted"].max(dim=1)[1]

        for pred in metric_dict["predicted"]:
            self.test_predicted.append(pred.tolist())

        for target in metric_dict["ground_truth"]:
            self.test_ground_truth.append(target.tolist())


    def get_report(self, mode='train'):
        """
        Get report at the end of training/testing

        Args:
        mode: Specify if train or test mode to generate report accordingly
        """


        if mode == "train":
        
            # Get max predicted values and convert to list for train
            train_predicted = np.array(self.train_predicted).argmax(axis=1).tolist()

            # GT values are converted to list
            train_gt = np.array(self.train_ground_truth).tolist()

            # Similar procedure for validation
            val_predicted = np.array(self.val_predicted).argmax(axis=1).tolist()
            val_gt = np.array(self.val_ground_truth).tolist()

            # Classification report used to generate precision, recall, f1-score across classes
            train_report = classification_report(train_gt, train_predicted, output_dict=True)
            val_report = classification_report(val_gt, val_predicted, output_dict=True)

            # Convert report to dictionary
            train_report = pd.DataFrame(train_report)
            val_report = pd.DataFrame(val_report)

            logger.info("Train Report: {} \n\n".format(train_report))
            logger.info("Validation Report: {}\n\n".format(val_report))

            # If upload is 1, upload precision, recall and f1scores
            if self.upload:
                train_scores = precision_recall_fscore_support(train_gt, train_predicted)
                val_scores = precision_recall_fscore_support(val_gt, val_predicted)

                wandb.log({
                            "precision@train" : np.mean(train_scores[0]).round(2),
                            "recall@train" : np.mean(train_scores[1]).round(2),
                            "f1-score@train" : np.mean(train_scores[2]).round(2),
                            "precision@val" : np.mean(val_scores[0]).round(2),
                            "recall@val" : np.mean(val_scores[1]).round(2),
                            "f1-score@val" : np.mean(val_scores[2]).round(2),
                            })

            return train_report, val_report

        elif mode == "test":

            # Similar procedure to above but on test data
            test_predicted = np.array(self.test_predicted).argmax(axis=1).tolist()
            test_gt = np.array(self.test_ground_truth).tolist()

            test_report = classification_report(test_gt, test_predicted, output_dict=True)
            test_report = pd.DataFrame(test_report)

            logger.info("Test Report: {} \n\n".format(test_report))
            return test_report



    def display(self):
        """
        Display averaged metrics over an entire epoch of training
        """

        # Average losses over batch
        self.metric_dict["loss@train"] = np.mean(self.train_losses)
        self.metric_dict["loss@val"] = np.mean(self.val_losses)


        train_predicted = np.array(self.train_predicted)
        train_gt = np.array(self.train_ground_truth)

        val_predicted = np.array(self.val_predicted)
        val_gt = np.array(self.val_ground_truth)

        # Get accuracies over all predictions over an epoch
        self.metric_dict["accuracy@train"] = np.mean(train_predicted.argmax(axis=1) == train_gt) * 100
        self.metric_dict["accuracy@val"] = np.mean(val_predicted.argmax(axis=1) == val_gt) * 100
        logger.info("Metrics {}".format(self.metric_dict))

        # If upload is set to 1, upload all the metrics along with confusion plot, ROC and sample image predictions
        if self.upload:
            self.metric_dict["Sample Predictions"] = []

            random_image_idx = np.random.choice(range(len(self.predicted_images)), 50).tolist()

            for idx, image in enumerate(self.predicted_images):
                if idx in random_image_idx:
                    self.metric_dict["Sample Predictions"].append(wandb.Image(image["Image"], \
                        caption="Predicted: {}, GT: {}".format(image["Predicted"], image["Ground Truth"])))

            self.metric_dict["roc"] = wandb.plots.ROC(val_gt, val_predicted, self.class_mapping)
            self.metric_dict["confusion_plot"] = wandb.sklearn.plot_confusion_matrix(val_gt, val_predicted.argmax(axis=1), self.class_mapping)

            wandb.log(self.metric_dict)



    def confusion_matrix_plot(self, class_mapping, test_split):
        """
        Show confusion matrix plot for collected metrics. Applied during testing

        Args:
        class_mapping: List mapping emotions to numeric labels
        test_split: Either PublicTest or PrivateTest
        """

        # Get class with maximum probability in the predictions and convert to list
        test_predicted = np.array(self.test_predicted).argmax(axis=1).tolist()
        test_gt = np.array(self.test_ground_truth).tolist()

        # Generate confusion matrix
        matrix = confusion_matrix(test_gt, test_predicted, normalize='true')
        
        # Convert to data frame with emotion labels
        df = pd.DataFrame(matrix, index = [i for i in class_mapping],
                        columns = [i for i in class_mapping])
        
        # Plot seaborn heatmap representing the Confusion matrix
        plt.figure(figsize = (10,8))
        plt.title('Confusion matrix for predictions on the {} split'.format(test_split))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        sns.heatmap(df, annot=True, cmap="YlGnBu")
        plt.show()
