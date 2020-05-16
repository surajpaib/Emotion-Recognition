import torch
import wandb
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Metrics:
    def __init__(self, upload=False):
        self.upload = upload
        self.reset()

    def reset(self):
        self.metric_dict = {}

        self.train_losses = []
        self.val_losses = []

        self.train_predicted = []
        self.train_ground_truth = []


        self.val_predicted = []
        self.val_ground_truth = []

        self.predicted_images = []


    def update_train(self, metric_dict):
        self.train_losses.append(metric_dict["loss"])

        for pred in metric_dict["predicted"]:
            self.train_predicted.append(pred.tolist())

        for target in metric_dict["ground_truth"]:
            self.train_ground_truth.append(target.tolist())


    def update_val(self, metric_dict):

        self.class_mapping = metric_dict["class_mapping"]
        pred_classes = metric_dict["predicted"].max(dim=1)[1]
        self.val_losses.append(metric_dict["loss"])

        for pred in metric_dict["predicted"]:
            self.val_predicted.append(pred.tolist())

        for target in metric_dict["ground_truth"]:
            self.val_ground_truth.append(target.tolist())


        for idx in range(metric_dict["image"].shape[0]):
            self.predicted_images.append({"Image": metric_dict["image"][idx], \
                 "Predicted": self.class_mapping[pred_classes[idx]],\
                      "Ground Truth": self.class_mapping[metric_dict["ground_truth"][idx]]})



    def display(self):
        self.metric_dict["loss@train"] = np.mean(self.train_losses)
        self.metric_dict["loss@val"] = np.mean(self.val_losses)


        train_predicted = np.array(self.train_predicted)
        train_gt = np.array(self.train_ground_truth)

        val_predicted = np.array(self.val_predicted)
        val_gt = np.array(self.val_ground_truth)


        self.metric_dict["accuracy@train"] = np.mean(train_predicted.argmax(axis=1) == train_gt) * 100
        self.metric_dict["accuracy@val"] = np.mean(val_predicted.argmax(axis=1) == val_gt) * 100


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


        logger.info("Metrics {}".format(self.metric_dict))
