from torch.utils.data import Dataset
import pandas as pd
import numpy as np 
import torch

from utils import normalize, preprocess

class FER2013Dataset(Dataset):    
    def __init__(self, file_path, usage='Training'):
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
        
        
        emotion, _, image = self.data.iloc[idx] #plus 1 to skip first row (column name)    

        emotion = int(emotion) 

        image = image.split(" ") 
        image = np.array(image, dtype=np.float32)

        image = preprocess(image)

        sample = {'image': image, 'emotion': emotion}
        
        return sample