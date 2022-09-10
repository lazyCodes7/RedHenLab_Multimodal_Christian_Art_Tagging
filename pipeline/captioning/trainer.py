import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import torch
import sys
sys.path.insert(1, '../')
from PIL import Image
import cv2
from nltk import tokenize
import nltk
from modules.captioning.transformer import DecoderBlock, TransformerBlock
from modules.captioning.dataset import ChristianArtDataset
from modules.captioning.vit import ArtDLClassifier
from tqdm import tqdm, trange
import random
import os
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw, ImageOps
import torch.nn.functional as F
from modules.feature_extraction.transform import Transform
from modules.captioning.vit_gpt2 import VisualGPT2Transformer
import pickle
from cocoeval import calculate_metrics

class Trainer:
    def __init__(
        self,
        device,
        batch_size=16, 
        data_dir = None,
        metadata_path = None,
        feature_extractor_path = None,
        epochs=10, 
        lr=1e-4,
        inference = False
    ):
        
        vit_model = ArtDLClassifier(2,2,2).to(device)
        vit_model.load_state_dict(torch.load(feature_extractor_path, map_location = device))
        self.t = Transform()
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.metadata_path = metadata_path
        self.save_model_on_epoch = True
        if(inference == False):
            self.train_set, self.val_set, self.test_set = self.generate_dataset(self.t.transform)

        self.model = VisualGPT2Transformer(
            encoder_model = vit_model,
            src_vocab_size = 197,
            target_vocab_size = 50257,
            embed_size = 768,
            num_layers = 6,
            forward_expansion = 4,
            heads = 8,
            dropout = 0.2,
            device = device,
            max_length = 1024
        ).to(self.device)
    
    
    def generate_dataset(self, transform):
        dataset = ChristianArtDataset(
            data_dir = self.data_dir,
            transform = transform,
            metadata_path = self.metadata_path
        )
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
        return train_set, val_set, test_set
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
    
    def generate_captions(
        self,
        model,
        tokenizer,
        image,
        prompt = " ",
        entry_count=10,
        entry_length=1024, #maximum number of words
        top_p=0.8,
        temperature=1.,
    ):

        model.eval()
        generated_num = 0
        generated_list = []

        filter_value = -float("Inf")
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

        with torch.no_grad():
            for entry_idx in range(entry_count):

                entry_finished = False
                if(entry_idx > 1):
                    prompt = random.choice(random_prompts)
                generated = torch.tensor(tokenizer.encode(f"<|{prompt}|>")).unsqueeze(0)
            

                for i in range(entry_length):
                    outputs = model(image.to(self.device), generated.to(self.device))
                    logits = outputs[0]
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = filter_value

                    next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                    generated = torch.cat((generated.to(self.device), next_token.to(self.device)), dim=1)

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


    
    def train(self):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        train_dataloader = DataLoader(self.train_set, batch_size=1, shuffle=True)
        loss=0
        accumulating_batch_count = 0

        print("Training...")
        for epoch in range(self.epochs):
            with tqdm(train_dataloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                for idx, (image, caption) in enumerate(tepoch):
                    image = image.to(self.device)
                    caption = caption.to(self.device)
                    outputs = self.model(image, caption)
                    outputs = outputs[0]
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = caption[..., 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)).to(self.device), shift_labels.view(-1).to(self.device))
                    tepoch.set_postfix(loss=loss.item())
                    loss.backward()

                    if (accumulating_batch_count % self.batch_size) == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    accumulating_batch_count += 1
                if self.save_model_on_epoch:
                    torch.save(
                        self.model.state_dict(),
                        f"captioning/runs/training/vit_gpt2-{epoch}.pt",
    
                    )
        
        print("Testing....")
        self.test()

    
    #Function to generate multiple sentences. Test data should be a dataframe
    def generate_all(self):
        generated_captions = []
        actual_captions = []

        for i in trange(len(self.test_set)):
            prompt = self.model.decoder.tokenizer.decode(self.test_set[i][1]).split("<|")[1].split("|>")[0]

            x = self.generate_captions(
                model = self.model, 
                tokenizer = self.model.decoder.tokenizer, 
                image = self.test_set[i][0].unsqueeze(0),
                entry_count = 5, 
                prompt = prompt,
                entry_length = self.test_set[i][1].shape[0]
            )
            generated_captions.append(x)
            actual_captions.append(self.model.decoder.tokenizer.decode(self.test_set[i][1]))

        return generated_captions, actual_captions
    
    def test(self):
        actual_coco = {}
        actual_coco['annotations'] = []
        generated_coco = {}
        generated_coco['annotations'] = []
        generated_captions, actual_captions = self.generate_all()
        processed_actual_captions = []
        processed_generated_captions = []
        for i in range(len(actual_captions)):
            processed_actual_captions.append(actual_captions[i].split("<|")[1].split("|>")[-1])
            temp_gen = []
            for j in range(len(generated_captions[i])):
                temp_gen.append(generated_captions[i][j].split("<|")[1].split("|>")[-1])
            processed_generated_captions.append(temp_gen)
        
        for i in range(len(actual_captions)):
            ac_coco_item = {'image_id': i, 'caption' : processed_actual_captions[i]}
            for j in range(len(processed_generated_captions[i])):
                gen_coco_item = {'image_id': i, 'caption' : processed_generated_captions[i][j]}
                generated_coco['annotations'].append(gen_coco_item)
            actual_coco['annotations'].append(ac_coco_item)

        rng = range(len(actual_captions))
        output = {"rng":rng, "actual_coco": actual_coco, "generated_coco": generated_coco}
        #results = calculate_metrics(rng, generated_coco, actual_coco)

        print("Saving Outputs.........")
        current_files = os.listdir('captioning/runs/training')
        if(len(current_files) == 0):
            filehandler = open(b"captioning/runs/training/outputs.pth","wb")
            filename = "captioning/runs/training/outputs.pth"
        else:
            filehandler = open("captioning/runs/training/outputs{}.pth".format(len(current_files)),"wb")
            filename = "captioning/runs/inferences/outputs{}.pth".format(len(current_files))
        
        pickle.dump(output,filehandler)
        print("Results saved successfully at {}".format(os.getcwd() + "/" + filename))
