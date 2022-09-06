import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import urllib
import os
class Generator:
    def __init__(self, csv_path, data_dir = None, force = False):
        self.metadata_df = pd.read_csv(csv_path)
        self.files_list = self.metadata_df['IMAGE_NAME']
        self.links = self.metadata_df['IMAGE_LINK']
        self.already_downloaded = os.listdir(data_dir)
        self.data_dir = data_dir
        self.force = force
        
    
    def generate(self):
        log_once = True
        for i, items in enumerate(tqdm(self.metadata_df.iterrows())):
            items = items[1]
            if(i%500 == 0):
                log_once = True
            if(items['IMAGE_NAME'] in self.already_downloaded and self.force == False):
                if(log_once):
                    print("File has already been downloaded. Set --force if you still want to download this image")
                    log_once = False
                continue
            link = items['IMAGE_LINK']
            name = items['IMAGE_NAME']
            urllib.request.urlretrieve(link, self.data_dir + name)


def main(metadata_path, data_dir, force):
    generator = Generator(metadata_path, data_dir, force)
    generator.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--metadata_path', type=str, required=True)
    parser.add_argument('-d', '--data_dir', type=str, required=True)
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    print(args.force)
    main(args.metadata_path, args.data_dir, args.force)



    
