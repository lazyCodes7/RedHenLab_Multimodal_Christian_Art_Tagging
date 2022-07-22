import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import pandas as pd
import numpy as np

class ChristianArtDataset(Dataset):
  def __init__(self, metadata_df, transform = None):
    self.metadata_df = metadata_df
    self.transform = transform
  
  def __getitem__(self, idx):
    record = self.metadata_df.iloc[idx]
    image = Image.open('Emile_Male_Dataset/Emile Male Pipeline/Data/Images/' + record['IMAGE_NAME']).convert('RGB')
    description = record['TITLE']
    if(self.transform):
      image = self.transform(image)
    

    return image, description
  

  def __len__(self):
    return len(self.metadata_df)
  