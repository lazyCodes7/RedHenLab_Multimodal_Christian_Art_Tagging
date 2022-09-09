import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from dataset import ArtDLDataset, DatasetCollector
from transform import Transform
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import argparse
from tqdm import tqdm
sys.path.insert(1, '../')
from modules.captioning.vit import ArtDLClassifier
class Trainer:
    def __init__(self, data_dir = None, labels_path = None, device = 'cpu', train_batch_size = 50, epochs = 10):
        tf = Transform()
        train_dataset = ArtDLDataset(
            data_dir = data_dir,
            transform = tf.transform,
            labels_path = labels_path,
            set_type = 'train'
        )

        test_dataset = ArtDLDataset(
            data_dir = data_dir,
            transform = tf.val_transform,
            labels_path = labels_path,
            set_type = 'test'
        )

        val_dataset = ArtDLDataset(
            data_dir = data_dir,
            transform = tf.val_transform,
            labels_path = labels_path,
            set_type = 'val'
        )
        self.train_loader = DataLoader(dataset = train_dataset, shuffle=True, batch_size = train_batch_size)
        self.test_loader = DataLoader(dataset = test_dataset, batch_size = 10)
        self.val_loader = DataLoader(dataset = val_dataset, batch_size = 10)
        self.device = device
        self.model = ArtDLClassifier(2,2,2).to(device)
        self.epochs = epochs
    
    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr = 0.0015, momentum = 0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            with tqdm(self.train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                # Setting the train mode
                self.model.train()
                train_loss = 0
                val_loss = 0
                for idx, (image, label) in enumerate(tepoch): 
                    image = image.to(self.device)
                    #print(image.shape)
                    label = label.to(self.device)

                    # Zeroing the gradients before re-computing them
                    optimizer.zero_grad()
                    outputs = self.model(image)

                    # Calculating the loss
                    loss = criterion(outputs['male_saint'], label[:,0]) + criterion(outputs['female_saint'], label[:,1]) + criterion(outputs['mary'], label[:,2])
                    tepoch.set_postfix(loss=loss.item())

                    train_loss += loss.item()

                    # Calculating the gradients == diff(loss w.r.t weights)
                    loss.backward()

                    # Updating the weights
                    optimizer.step()
            
            with torch.no_grad():
                self.model.eval()
                mary_score = 0
                male_saint_score = 0
                female_saint_score = 0
                for idx, (image, label) in enumerate(self.val_loader):
                    image = image.to(self.device)
                    label = label.to(self.device)
                    outputs = self.model(image)
                    male_saint_pred = outputs['male_saint'].argmax(dim = 1, keepdim = True)
                    female_saint_pred = outputs['female_saint'].argmax(dim = 1, keepdim = True)
                    mary_pred = outputs['mary'].argmax(dim = 1, keepdim = True)

                    # Updating scores and losses
                    male_saint_score += male_saint_pred.eq(label[:,0].view_as(male_saint_pred)).sum().item()
                    female_saint_score += female_saint_pred.eq(label[:,1].view_as(female_saint_pred)).sum().item()
                    mary_score+= mary_pred.eq(label[:,2].view_as(mary_pred)).sum().item()

                    loss = criterion(outputs['male_saint'], label[:,0]) + criterion(outputs['female_saint'], label[:,1]) + criterion(outputs['mary'], label[:,2])
                    val_loss += loss.item()
                
            print("=================================================")
            print("Epoch: {}".format(epoch+1))
            print("Validation Loss: {}".format(val_loss/len(self.val_loader)))
            print("Training Loss: {}".format(train_loss/len(self.train_loader)))
            print("Classificaton score for Mary: {}".format((mary_score)/len(self.val_loader)*10))
            print("Classificaton score for Male Saints: {}".format((male_saint_score)/len(self.val_loader)*10))
            print("Classificaton score for Female Saints: {}".format((female_saint_score)/len(self.val_loader)*10))

            
    def test(self, save_path):
        self.model.eval()
        self.model.load_state_dict(torch.load(save_path, map_location = self.device))
        with torch.no_grad():
            test_loss = 0
            mary_score = 0
            male_saint_score = 0
            female_saint_score = 0
            for idx, (image, label) in enumerate(self.test_loader):
                image = image.to(self.device)
                label = label.to(self.device)
                outputs = self.model(image)
                male_saint_pred = outputs['male_saint'].argmax(dim = 1, keepdim = True)
                female_saint_pred = outputs['female_saint'].argmax(dim = 1, keepdim = True)
                mary_pred = outputs['mary'].argmax(dim = 1, keepdim = True)

                # Updating scores and losses
                male_saint_score += male_saint_pred.eq(label[:,0].view_as(male_saint_pred)).sum().item()
                female_saint_score += female_saint_pred.eq(label[:,1].view_as(female_saint_pred)).sum().item()
                mary_score+= mary_pred.eq(label[:,2].view_as(mary_pred)).sum().item()

                loss = criterion(outputs['male_saint'], label[:,0]) + criterion(outputs['female_saint'], label[:,1]) + criterion(outputs['mary'], label[:,2])
                test_loss += loss.item()
            
        print("=================================================")
        print("Validation Loss: {}".format(val_loss/len(self.test_loader)))
        print("Classificaton score for Mary: {}".format((mary_score)/len(self.test_loader)*10))
        print("Classificaton score for Male Saints: {}".format((male_saint_score)/len(self.test_loader)*10))
        print("Classificaton score for Female Saints: {}".format((female_saint_score)/len(self.test_loader)*10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--train_batch_size', type = int, required = False, default = 50)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--device', type=str, required=False, default = 'cpu')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('-p', '--model_path', type=str, required=False, default='')
    parser.add_argument('-c', '--collect', action='store_true')

    args = parser.parse_args()
    if(args.collect):
        collector = DatasetCollector(args.data_dir)
        collector.unzip()

    trainer = Trainer(
        data_dir = args.data_dir + "DEVKitArt/JPEGImages/", 
        labels_path = args.data_dir + "DEVKitArt/ImageSets/Main/info.csv",
        device = args.device, 
        train_batch_size = args.train_batch_size
    )
    if(args.train):
      trainer.train()
      torch.save(trainer.model.state_dict(), 'artDL.pt')
    
    else:
      trainer.test(args.model_path)


    

