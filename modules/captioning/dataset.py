import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import torch
from PIL import Image
import cv2
import torchvision.models as models
from nltk import tokenize
import nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import random 
nltk.download('punkt')

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
        '''
        About:
            Instantiates the dataset class for iterating over art collections
        
        Inputs:
            data_dir - Path to take images from
            transform - torchvision.transforms - preprocessing/augmentation techniques to apply
            gpt2_type - gpt2 tokenizer to use
            max_length - truncation length to truncate sentences

        
        Methods:
            1. __len__(self)
                
                Returns the length of the dataset
            
            2. __getitem__(self, item)

                Get an item from the dataset

        
        Example:
            from dataset import ChristianArtDataset
            dataset = ChristianArtDataset(
                data_dir = 'Images/'
                transform = transform
            )
            print(dataset[0])

        '''  
        self.metadata_df = pd.read_csv('/content/drive/MyDrive/Emile Male Pipeline/Data/metadata_v2.csv')
        self.metadata_df = self.metadata_df[self.metadata_df['TITLE'].isnull() == False]
        self.metadata_df.reset_index(inplace = True, drop = True)
        self.metadata_df['TITLE_TRUNCATED'] = self.metadata_df['TITLE'].apply(lambda x : " ".join(tokenize.word_tokenize(x)[:512]))
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.descriptions = []
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
        for i in range(len(self.metadata_df)):
          self.descriptions.append(torch.tensor(
                self.tokenizer.encode(f"<|{random.choice(self.random_prompts)}|>{self.metadata_df.iloc[i]['TITLE_TRUNCATED'][:max_length]}<|endoftext|>")
            ))               

        self.desc_count = len(self.descriptions)
        
    def __len__(self):
        # Return length of the dataset
        return self.desc_count

    def __getitem__(self, item):
        # Retrieves a sample from the dataset
        image = self.data_dir + self.images[item]
        image = cv2.imread(image)
        image = Image.fromarray(image).convert("RGB")
        
        if (self.transform):
            image = self.transform(image)
        return image, self.descriptions[item]

class ChristianArtDataset_Desc(Dataset):  

    def __init__(self, data_dir, transform = None, gpt2_type="gpt2", max_length=1024):
        '''
        About:
            Instantiates the dataset class for iterating over art collections with provided descriptions
        
        Inputs:
            data_dir - Path to take images from
            transform - torchvision.transforms - preprocessing/augmentation techniques to apply
            gpt2_type - gpt2 tokenizer to use
            max_length - truncation length to truncate sentences
        
        Methods:
            1. __len__(self)
                
                Returns the length of the dataset
            
            2. __getitem__(self, item)

                Get an item from the dataset
        
        
        Example:
            from dataset import ChristianArtDataset_Desc
            dataset = ChristianArtDataset_Desc(
                data_dir = 'Images/'
                transform = transform
            )
            print(dataset[0])

        '''  
        self.metadata_df = pd.read_csv('/content/drive/MyDrive/Emile Male Pipeline/Data/metadata_v2.csv')
        self.metadata_df = self.metadata_df[self.metadata_df['DESCRIPTION'].isnull() == False]
        self.metadata_df.reset_index(inplace = True, drop = True)
        self.metadata_df['DESCRIPTION_TRUNCATED'] = self.metadata_df['DESCRIPTION'].apply(lambda x : " ".join(tokenize.word_tokenize(x)[:1024]))
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.descriptions = []
        self.images = self.metadata_df["IMAGE_NAME"]
        self.data_dir = data_dir
        self.transform = transform

        for i in range(len(self.metadata_df)):
          self.descriptions.append(torch.tensor(
                self.tokenizer.encode(f"<|{self.metadata_df.iloc[i]['TITLE']}|>{self.metadata_df.iloc[i]['DESCRIPTION_TRUNCATED'][:max_length]}<|endoftext|>")
            ))               

        self.desc_count = len(self.descriptions)
        
    def __len__(self):
        return self.desc_count

    def __getitem__(self, item):
        # Retrieves a sample from the dataset
        image = self.data_dir + self.images[item]
        image = cv2.imread(image)
        image = Image.fromarray(image).convert("RGB")
        
        if (self.transform):
            image = self.transform(image)
        return image, self.descriptions[item]
    
