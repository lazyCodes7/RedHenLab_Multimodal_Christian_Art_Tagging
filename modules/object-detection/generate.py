from dataset import ChristianArtDataset
from transform import Transform
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import argparse
import torch
import time
import os
from PIL import Image
from torchvision.ops import nms
def generate_anchors(images, model, thresh = 0.4):
    st = time.time()
    model.eval()

    results = model(images.float())
    confidences = []
    bboxes = []
    anchor_labels = []
    for result in results:
        scores, transformed_anchors, classification = result['scores'], result['boxes'], result['labels']
        confidences.append(scores.cpu().detach().numpy())
        bboxes.append(transformed_anchors.cpu().detach().numpy())
        anchor_labels.append(classification.cpu().detach().numpy())
        
    
    anchor_df = pd.DataFrame({"bboxes" : bboxes, "confidences": confidences, "labels":anchor_labels})
    et=time.time()
    print("\n Total Time - {}\n".format((et - st)))

    return anchor_df

def generate_all(batch_size = 20, threshold = 0.4):
    transform = Transform()
    device = "cpu"
    
    metadata_df = pd.read_csv('Emile_Male_Dataset/Emile Male Pipeline/Data/metadata.csv')
    art_dataset = ChristianArtDataset(metadata_df, transform = transform.transform)
    try:
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.to(device)
        model.load_state_dict(torch.load('frcnn_iconart_v2.pt', map_location=device))        
    except Exception as e:
        print(e)
    loader = DataLoader(art_dataset, batch_size = batch_size, shuffle = True)
    anchors = pd.DataFrame()
    for idx, (images, label) in enumerate(loader):
        images = images.to(device)
        generated_anchors = generate_anchors(images, model, threshold)
        anchors.append(generated_anchors)
    return anchors

def generate_image_dir(image_dir, threshold = 0.4):
    transform = Transform()
    device = "cpu"
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    images = []
    for image_link in os.listdir(image_dir):
        image = transform.transform(Image.open(image_dir + "/" + image_link))
        images.append(image)
        
    images = torch.stack(images, dim = 0)

    anchors = generate_anchors(images, model, threshold)
    return anchors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, required = False, default = 20)
    parser.add_argument('--threshold', type=int, required=False, default = 0.4)
    parser.add_argument('--image_dir', type=str, required=False)
    parser.add_argument('--save_dir', type=str, default = '', required = False)
    args = parser.parse_args()
    print(args.image_dir)
    if(args.image_dir!=None):
        anchors = generate_image_dir(args.image_dir, args.threshold)
        anchors.to_csv(args.save_dir + "anchors.csv")
    
    else:
        anchors = generate_all(args.batch_size, args.threshold)
        anchors.to_csv(args.save_dir + "anchors.csv")

    

    







