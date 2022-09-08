
import gdown
from zipfile import ZipFile
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as FT
class DatasetCollector:
	## Import from drive
	def __init__(self):
		url = "https://drive.google.com/uc?id=16FK1YnHPhGqCHf_EpovzcH0v90yXcCer"
		output = 'artDL.zip'
		gdown.download(url, output, quiet=False)

	def unzip(self, path):
		with ZipFile(path, 'r') as zipObj:
			# Extract all the contents of zip file in current directory
			zipObj.extractall()

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
    image = Image.open(self.data_dir + "/" + filename + ".jpg")

    # Applying transforms if any
    if(self.transform!=None):
      image = self.transform(image)
    
    # Getting the label 
    image_label = self.labels_df[self.labels_df['item'] == filename].values.squeeze()[2:20].argmax()
    #print(image_label)
    
    return (image, image_label)

  def __len__(self):
    return len(self.img_names)