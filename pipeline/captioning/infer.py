from trainer import Trainer
import argparse
import torch
from PIL import Image
from tqdm import tqdm, trange
import os
import pickle
import random
def infer(
    device, 
    image_dir, 
    metadata_path, 
    feature_extractor_path, 
    model_path,
    entry_count,
    entry_length
):
    prompts = [
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
    trainer = Trainer(
        device = device,
        metadata_path = metadata_path,
        feature_extractor_path = feature_extractor_path,
        inference = True
    )
    trainer.model.load_state_dict(torch.load(model_path, map_location = device))
    images = os.listdir(image_dir)

    generated_captions = []
    for i in trange(len(images)):
        prompt = random.choice(prompts)
        image = Image.open(image_dir + images[i])
        image = trainer.t.val_transform(image)
        x = trainer.generate_captions(
            model = trainer.model, 
            tokenizer = trainer.model.decoder.tokenizer, 
            image = image.unsqueeze(0),
            entry_count = entry_count, 
            prompt = prompt,
            entry_length = entry_length
        )
        generated_captions.append(x)
    

    generated_coco = {}
    generated_coco['annotations'] = []
    processed_generated_captions = []

    for i in range(len(generated_captions)):
        temp_gen = []
        for j in range(len(generated_captions[i])):
            temp_gen.append(generated_captions[i][j].split("<|")[1].split("|>")[-1])
        processed_generated_captions.append(temp_gen)

    
    for i in range(len(images)):
        for j in range(len(processed_generated_captions[i])):
            gen_coco_item = {'image_id': images[i], 'caption' : processed_generated_captions[i][j]}
            generated_coco['annotations'].append(gen_coco_item)
    
    print("Saving generated_captions.........")
    current_files = os.listdir('captioning/runs/inferences')
    if(len(current_files) == 0):
        filehandler = open(b"captioning/runs/inferences/inference.pth","wb")
        filename = "captioning/runs/inferences/inference.pth"
    else:
        filehandler = open("captioning/runs/inferences/inference{}.pth".format(len(current_files)),"wb")
        filename = "captioning/runs/inferences/inference{}.pth".format(len(current_files))

        
    pickle.dump(generated_coco,filehandler)
    print("Results saved successfully at {}".format(filename))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, required=False, default = '../modules/captioning/metadata_v2.csv')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--feature_extractor_path', type=str, required=True)
    parser.add_argument('-d', '--device', type = str, default = 'cuda')
    parser.add_argument('--captioning_model_path', type = str, default = '')
    parser.add_argument('--entry_count', type = int, default = 2)
    parser.add_argument('--entry_length', type = int, default = 30)



    args = parser.parse_args()
    infer(
        args.device,
        args.image_dir,
        args.metadata_path,
        args.feature_extractor_path,
        args.captioning_model_path,
        args.entry_count,
        args.entry_length
    )



