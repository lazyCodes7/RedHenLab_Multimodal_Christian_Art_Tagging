import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import torch
from PIL import Image
import cv2
import torchvision.models as models


import torchvision.transforms as transforms
import torchvision.transforms.functional as FT 

# Util class to apply padding to all the images
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return FT.pad(image, padding, 0, 'constant')


class ChristianArtDataset(Dataset):  
    def __init__(self, data_dir, transform = None, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.descriptions = []
        self.metadata_df = metadata_df
        self.images = self.metadata_df["IMAGE_NAME"]
        self.data_dir = data_dir
        self.transform = transform
        self.random_prompts = [
            "Describe the painting.",
            "What’s going on in this artwork?",
            "What title would you give this artwork?",
            "What symbols do you notice in the artwork?",
            "What is the subject matter?",
            "Describe the agents in painting.",
            "Describe the icons in painting.",
            "Best title for this painting",
            "Caption this painting",
            "Describe the artwork",
            "What is this painting about?",
            " "
        ]
        for i in range(len(metadata_df)):
          self.descriptions.append(torch.tensor(
                self.tokenizer.encode(f"<|{random.choice(self.random_prompts)}|>{metadata_df.iloc[i]['TITLE_TRUNCATED'][:max_length]}<|endoftext|>")
            ))               

        self.desc_count = len(self.descriptions)
        
    def __len__(self):
        return self.desc_count

    def __getitem__(self, item):
        image = self.data_dir + self.images[item]
        image = cv2.imread(image)
        image = Image.fromarray(image).convert("RGB")
        
        if (self.transform):
            image = self.transform(image)
        return image, self.descriptions[item]

class ChristianArtDataset_Desc(Dataset):  
    def __init__(self, data_dir, transform = None, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.descriptions = []
        self.metadata_df = metadata_df
        self.images = self.metadata_df["IMAGE_NAME"]
        self.data_dir = data_dir
        self.transform = transform

        for i in range(len(metadata_df)):
          self.descriptions.append(torch.tensor(
                self.tokenizer.encode(f"<|{metadata_df.iloc[i]['TITLE']}|>{metadata_df.iloc[i]['DESCRIPTION_TRUNCATED'][:max_length]}<|endoftext|>")
            ))               

        self.desc_count = len(self.descriptions)
        
    def __len__(self):
        return self.desc_count

    def __getitem__(self, item):
        image = self.data_dir + self.images[item]
        image = cv2.imread(image)
        image = Image.fromarray(image).convert("RGB")
        
        if (self.transform):
            image = self.transform(image)
        return image, self.descriptions[item]
    
