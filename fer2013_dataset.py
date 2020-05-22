from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

from utils.utils import normalize, preprocess

class FER2013Dataset(Dataset):
    def __init__(self, file_path, usage='Training'):
        """
        FER2013 Dataset Class inherited from Pytorch dataset.

        Args:
        file_path: Location of the csv file with the FER data
        usage: Select which label to use for the dataset, either Training, PrivateTest or PublicTest

        """



        self.file_path = file_path
        self.usage = usage
        
        # Read the csv file and extract entries matching the usage criterion
        self.data = pd.read_csv(self.file_path)
        self.data = self.data[self.data["Usage"] == self.usage]

        # Set dataset length for loaders
        self.total_images = len(self.data)

    def __len__(self):
        return self.total_images


    def get_summary_statistics(self):
        """
        Get summary statistics for the dataset. This is used to compute the weighted loss scheme.
        """
        summary = {}
        # Normalized frequencies of the classes in the dataset are obtained. 
        summary["class_occurences"] = self.data["emotion"].value_counts(normalize=True, sort=False).values
        return summary

    def get_class_mapping(self):
        """
        Get list mapping numeric labels to emotions
        """
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        return self.classes

    def __getitem__(self, idx):
        """
        Load single item from the dataset
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get a particular row from the dataframe
        emotion, image, _ = self.data.iloc[idx]
        emotion = int(emotion)

        # Parse image from string and convert to a numpy array
        image = image.split(" ")
        image = np.array(image, dtype=np.float32)

        # Preprocess the image (Normalization and reshaping)
        image = preprocess(image)

        # Return dict with image and emotion 
        sample = {'image': image, 'emotion': emotion}

        return sample