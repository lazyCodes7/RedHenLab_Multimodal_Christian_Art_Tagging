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

def generate(
    model,
    tokenizer,
    prompt,
    image,
    device,
    entry_count=10,
    entry_length=1024, #maximum number of words
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
                logits = 0.8*outputs[0] + 0.4*outputs[1]
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



def train(
    dataset, model, device, criterion,
    batch_size=16, epochs=10, lr=1e-4,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=False,
):
    acc_steps = 100
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
          tepoch.set_description(f"Epoch {epoch+1}")
          for idx, (image, caption) in enumerate(tepoch):
              image = image.to(device)
              caption = caption.to(device)
              outputs = model(image, caption)
              outputs = outputs[0]
              shift_logits = outputs[..., :-1, :].contiguous()
              shift_labels = caption[..., 1:].contiguous()
              loss = criterion(shift_logits.view(-1, shift_logits.size(-1)).to(device), shift_labels.view(-1).to(device))
              tepoch.set_postfix(loss=loss.item())
              loss.backward()

              if (accumulating_batch_count % batch_size) == 0:
                  optimizer.step()
                  optimizer.zero_grad()
                  model.zero_grad()

              accumulating_batch_count += 1
          if save_model_on_epoch:
              torch.save(
                  model.state_dict(),
                  os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
              )
    return model

def run(train, dataset, device, write_every = 10):
    transform=transforms.Compose([
        # Padding is done to ensure resolution is same after resizing
        transforms.Resize((224,224)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5)
    ])


    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor()
        
    ])
    if(dataset == 'desc'):
        dataset = ChristianArtDataset(data_dir = '/content/drive/MyDrive/Emile Male Pipeline/Data/Images/', transform = transform)
    else:
        dataset = ChristianArtDataset(data_dir = '/content/drive/MyDrive/Emile Male Pipeline/Data/Images/', transform = transform)

    vit_model = ArtDLClassifier(2,2,2).to(device)
    vit_model.load_state_dict(torch.load('/content/drive/MyDrive/Emile Male Pipeline/Models/vit_224x224_v1.pt', map_location = device))
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

    criterion = nn.CrossEntropyLoss()
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    n = len(dataset)

    n_test = int( n * .2) 
    n_val = int(n * .2) # number of test/val elements
    n_train = n - 2 * n_test


    train_idx = idx[:n_train]
    val_idx = idx[n_train:(n_train + n_test)]
    test_idx = idx[(n_train + n_test):]


    train_set, val_set, test_set = data.random_split(dataset, (n_train, n_val, n_test))

    if(train):
        train(train_set, model, device = device, criterion = criterion)
    
    else:
        path = '/content/drive/MyDrive/CIFAR-10 Models/vit_base_title_224x224.pt'
        model.load_state_dict(torch.load(path, map_location = device))
        bleu_score = 0
        generated_titles = []
        for i in range(10):
            x = generate(model.to(device), model.decoder.tokenizer, " ", test_set[i][0].unsqueeze(0), entry_count=1, entry_length = len(test_set[i][1].squeeze()), device = device)
            given_sentence = model.decoder.tokenizer.decode(test_set[i][1])
            if(i%write_every == 0):
                img = test_set[i][0].permute(1,2,0).numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                generated = "".join(x)
                print(generated)
                img = ImageOps.expand(img, border=20, fill=(255,255,255))
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("arial.ttf", 24)
                draw.text((0,0),generated,(0,255,255),font=font)
                img.save('examples/sample-{}.jpg'.format(i))
            bleu_score += sentence_bleu(given_sentence, x)
            generated_titles.append(x)
        
        print("===============================================")
        print(bleu_score)
        return generated_lyrics

if __name__ == "__main__":
    run(False, '', 'cpu')
        

