import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F
import torchvision.transforms.functional as FT 
from zipfile import ZipFile
import gdown
import os
import torch
from PIL import Image
class DatasetCollector:
	## Import from drive
	def __init__(self, path):
		self.path = path
		print("Downloading dataset.....")
		url = "https://drive.google.com/uc?id=16FK1YnHPhGqCHf_EpovzcH0v90yXcCer"
		output = 'artDL.zip'
		if(output in os.listdir(path)):
			print("Zip file already downloaded. Proceeding to extraction.")
		else:
			output = path + 'artDL.zip'
			gdown.download(url, output, quiet=False)

	def unzip(self):
		with ZipFile(self.path + 'artDL.zip', 'r') as zipObj:
			# Extract all the contents of zip file in current directory
			zipObj.extractall(self.path)

		info_df = pd.read_csv(self.path + 'DEVKitArt/ImageSets/Main/info.csv')
		info_df['MALE_SAINT_PRESENT'] = info_df[
			[
				"11H(ANTONY ABBOT)",
				"11H(ANTONY OF PADUA)",
				"11H(AUGUSTINE)",
				"11H(DOMINIC)", 	
				"11H(FRANCIS)", 	
				"11H(JEROME)", 	
				"11H(JOHN THE BAPTIST)", 	
				"11H(JOHN)", 
				"11H(PAUL)", 
				"11H(PETER)", 
				"11H(SEBASTIAN)" ,
				"11H(STEPHEN)",
				"John Baptist - Child",
				"John Baptist - Dead"
			]
		].sum(axis = 1)


		info_df['FEMALE_SAINT_PRESENT'] = info_df[
			[
				"11HH(BARBARA)",
				"11HH(CATHERINE)", 	
				"11HH(MARY MAGDALENE)"
			]
		].sum(axis = 1)


		info_df['MARY_PRESENT'] = info_df[
			[
				"11F(MARY)",
			]
		].sum(axis = 1)

		info_df[info_df["FEMALE_SAINT_PRESENT"]>0] = 1
		info_df[info_df["MALE_SAINT_PRESENT"]>0] = 1
		info_df.to_csv(self.path + 'DEVKitArt/ImageSets/Main/info.csv', index = False)
class ArtDLDataset(Dataset):
  def __init__(self, data_dir = None, transform = None, labels_path = None, set_type = 'train'):

    # Setting the inital_dir to take images from
    self.data_dir = data_dir

    # Setting up the transforms
    self.transform = transform

    # Label path to reads labels_csv from
    self.labels_path = labels_path
    labels_df = pd.read_csv(self.labels_path)

    # Filtering df based on set type
    self.labels_df = labels_df[labels_df['set'] == set_type]
    self.img_names = list(self.labels_df['item'])

  def __getitem__(self, idx):

    # Getting the filename based on idx
    filename = self.img_names[idx]
    #print(filename)

    # Reading using PIL
    image = Image.open(self.data_dir + filename + ".jpg")

    # Applying transforms if any
    if(self.transform!=None):
      image = self.transform(image)
    
    # Getting the label 
    image_label = self.labels_df[self.labels_df['item'] == filename].values.squeeze()[21:]
    image_label = image_label.astype(np.uint8)
    #print(image_label)

    return image, torch.tensor(image_label)
    

  def __len__(self):
    return len(self.img_names)

    
