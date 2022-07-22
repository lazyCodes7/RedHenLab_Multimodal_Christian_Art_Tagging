from models import ArtDLClassifier
import torch.nn as nn
from transform import Transform
import os
import pickle
import argparse
from PIL import Image
import torch

def extract(image_dir, device, load_path):
    images = []
    transform = Transform()
    for image in os.listdir(image_dir):
        image = Image.open(image_dir + "/" + image)
        image = transform.val_transform(image)
        images.append(image)
    
    images = torch.stack(images, dim = 0)
    model = ArtDLClassifier(num_classes = 19).to(device)
    module = model.net[4][0].conv1
    key = str(module)
    ac_handler = module.register_forward_hook(model.get_activations(key))
    images = images.to(device)
    model.load_state_dict(torch.load(load_path, map_location = device))
    out_features = model(images)
    ac_handler.remove()
    temp = model.activations[key]
    model.activations = {}
    return temp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--image_dir', type=str, required=True)
    parser.add_argument('-de','--device', type=str, required=False, default = 'cpu')
    parser.add_argument('-m', '--model_path', type=str, required=False, default='')
    args = parser.parse_args()
    extraction = extract(args.image_dir, args.device, args.model_path)
    print(extraction.shape)
    filename = 'extracted_features.pth'
    pickle.dump(extraction, open(filename, 'wb'))
    
    