from torch.utils.data import Dataset
import pandas as pd
import numpy as np 
import torch

from utils import normalize

class FER2013Dataset(Dataset):
    """Face Expression Recognition Dataset"""
    
    def __init__(self, file_path, usage='Training'):
        """
        Args:
            file_path (string): Path to the csv file with emotion, pixel & usage.
        """
        self.file_path = file_path 
        self.usage = usage
        self.data = pd.read_csv(self.file_path)

        self.data = self.data[self.data[" Usage"] == self.usage]

        self.total_images = len(self.data) # Ignore header row

    def __len__(self):  
        return self.total_images


    def get_summary_statistics(self):
        """
        Get summary statistics for the dataset. 
        """
        summary = {}
        summary["class_occurences"] = self.data["emotion"].value_counts(normalize=True, sort=False).values
        return summary

    def get_class_mapping(self):
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        return self.classes
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        emotion, _, img = self.data.iloc[idx] #plus 1 to skip first row (column name)    

        emotion = int(emotion) 
        img = img.split(" ") 
        img = np.array(img, dtype=np.uint8)
        img = img.reshape(48,48) 
        img = normalize(img)

        sample = {'image': img, 'emotion': emotion}
        
        return sample