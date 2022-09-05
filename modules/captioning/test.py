import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import torch
from PIL import Image
import cv2
from nltk import tokenize
import nltk
from dataset import ChristianArtDataset, ChristianArtDataset_Desc
from vit_gpt2 import VisualGPT2Transformer
from vit import ArtDLClassifier
from tqdm import tqdm, trange
import random
import os
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw, ImageOps
import torch.nn.functional as F
import requests
import pickle
def generate(
    model,
    tokenizer,
    prompt,
    image,
    device,
    entry_count=10,
    entry_length=60, #maximum number of words
    top_p=0.8,
    temperature=1.
):
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(image.to(device), generated.to(device))
                logits = outputs[0]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated.to(device), next_token.to(device)), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().cpu().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
              output_list = list(generated.squeeze().cpu().numpy())
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
              generated_list.append(output_text)
                
    return generated_list
def run(device, write_every = 10, image_dir = None):

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        
    ])
    images = []
    true_descriptions = []
    generated_descriptions = []
    if(image_dir == None):
        df = pd.read_csv('metadata_v2.csv')
        for i in range(30):
            image = val_transform(Image.open(requests.get(df.iloc[i]['IMAGE_LINK'], stream = True).raw))
            
            images.append(image)
        image_links = list(df['IMAGE_LINK'][:len(images)])

        images = torch.stack(images, dim = 0)
    else:
        for image_loc in os.listdir(image_dir):
            image = val_transform(Image.open(image_dir + "/" + image_loc).convert('RGB'))
            images.append(image)
        images = torch.stack(images, dim = 0)
    

    vit_model = ArtDLClassifier(2,2,2).to(device)
    vit_model.load_state_dict(torch.load('vit_224x224_v1.pt', map_location = device))
    model = VisualGPT2Transformer(
        encoder_model = vit_model,
        src_vocab_size = 197,
        target_vocab_size = 50257,
        src_pad_idx = 0,
        target_pad_idx = 50256,
        embed_size = 768,
        num_layers = 6,
        forward_expansion = 4,
        heads = 8,
        dropout = 0.2,
        device = device,
        max_length = 1024
    ).to(device)
    random_prompts = [
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
    path = 'vit_base_title_224x224.pt'
    model.load_state_dict(torch.load(path, map_location = device))
    bleu_score = 0
    for i in range(len(images)):
        given_sentence = None
        print("Generated sentence")
        x = generate(model.to(device), model.decoder.tokenizer, random.choice(random_prompts), images[i].unsqueeze(0), entry_count=1, entry_length = 60, device = device)
        print(x)
        if(image_dir == None):
            true_descriptions.append(df.iloc[i]['TITLE'])

        generated_descriptions.append(x)
        #current_bleu_score = sentence_bleu(df.iloc[i]['TITLE'], x)
        #print(current_bleu_score)
        #bleu_score += current_bleu_score
    
    #print("===============================================")
    #print(bleu_score/len(images))


    if(image_dir == None):
        result_df = pd.DataFrame({"image_links": image_links, "true_descriptions" : true_descriptions, "generated_descriptions": generated_descriptions})
        result_df.to_csv('inference.csv', index = False)
        
    else:
        result_df = pd.DataFrame({"generated_descriptions": generated_descriptions})
        result_df.to_csv('inference.csv', index = False)



if __name__ == "__main__":
    run('cpu', image_dir = 'image_dir')