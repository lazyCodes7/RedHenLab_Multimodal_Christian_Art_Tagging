
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from utils.collector import ArtDLDataset
from dataset import Transform
from models import ArtDLClassifier
import torch
import torch.optim as optim
class Trainer:
    def __init__(self, data_dir = None, labels_path = None, device, train_batch_size = 50):
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
        self.train_loader = DataLoader(dataset = train_dataset, shuffle=True, train_batch_size = 50)
        self.test_loader = DataLoader(dataset = test_dataset, batch_size = 1)
        self.val_loader = DataLoader(dataset = val_dataset, batch_size = 10)
        self.device = device
        self.model = ArtDLClassifier(num_classes = 19).to(device)
            
    def train(self):
        optimizer = optim.SGD(clf.trainable_params(), lr = 0.01, momentum = 0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            # Setting the train mode
            self.model.train()
            train_loss = 0
            val_loss = 0
            for idx, (image, label) in enumerate(self.train_loader):
                image = image.to(device)
                #print(image.shape)
                label = label.to(device)

                # Zeroing the gradients before re-computing them
                optimizer.zero_grad()
                outputs = self.model(image).squeeze()

                # Calculating the loss
                loss = criterion(outputs, label)
                train_loss += loss.item()

                # Calculating the gradients == diff(loss w.r.t weights)
                loss.backward()

                # Updating the weights
                optimizer.step()
                
                with torch.no_grad():
                    self.model.eval()
                    val_score = 0
                    for idx, (image, label) in enumerate(self.val_loader):
                        image = image.to(device)
                        label = label.to(device)
                        outputs = self.model(image).squeeze()

                        # Getting the predictions
                        pred = outputs.argmax(dim = 1, keepdim = True)

                        # Updating scores and losses
                        val_score += pred.eq(label.view_as(pred)).sum().item()
                        loss = criterion(outputs, label)
                        val_loss += loss.item()
                
            print("=================================================")
            print("Epoch: {}".format(epoch+1))
            print("Validation Loss: {}".format(val_loss/len(self.val_loader)))
            print("Training Loss: {}".format(train_loss/len(self.train_loader)))
            print("Validation Accuracy: {}".format((val_score)/len(self.val_loader)*10))
            
    def test(self, save_path):
        self.model.eval()
        self.model.load_state_dict(torch.load(save_path, map_location = self.device))
        test_score = 0
        img_count = 0
        images = []
        preds = []
        labels = []
        for idx, (image, label) in enumerate(self.test_loader):
            image = image.to(device)
            label = label.to(device)
            outputs = model(image).squeeze()
            #print(outputs)
            pred = outputs.argmax()
            preds.append(pred.item())
            labels.append(label.item())
            #print(pred)
            if(pred == label):
            if(test_score<10):
                images.append(image)
            test_score+=1

        print("Test Accuracy {:.3f}".format(test_score/len(test_loader)))

        return preds, labels, images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_batch_size', type = int, required = False, default = 50)
    parser.add_argument('-d','--data_dir', type=str, required=True)
    parser.add_argument('-de','--device', type=str, required=False, default = 'cpu')
    parser.add_argument('-tr', '--train', type=bool, required=False, default=False)
    parser.add_argument('-m', '--model_path', type=str, required=False, default='')
    args = parser.parse_args()

    trainer = Trainer(args.data_dir, args.device, args.train_batch_size)
    if(args.train):
      trainer.train()
      trainer.model.state_dict('artDL.pt')
    
    else:
      trainer.test(args.model_path)


    

