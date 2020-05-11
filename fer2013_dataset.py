from torch.utils.data import Dataset
import pandas as pd
import numpy as np 
import torch

class FER2013Dataset(Dataset):
    """Face Expression Recognition Dataset"""
    
    def __init__(self, file_path):
        """
        Args:
            file_path (string): Path to the csv file with emotion, pixel & usage.
        """
        self.file_path = file_path
        self.classes = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral') # Define the name of classes / expression
        
        self.data = pd.read_csv(self.file_path)
        self.total_images = len(self.data) # Ignore header row

    def __len__(self):  
        return self.total_images
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        emotion, img = self.data.iloc[idx] #plus 1 to skip first row (column name)    

        emotion = int(emotion) 
        img = img.split(" ") 
        img = np.array(img, dtype=np.uint8)
        img = img.reshape(48,48) 

        sample = {'image': img, 'emotion': emotion}
        
        return sample