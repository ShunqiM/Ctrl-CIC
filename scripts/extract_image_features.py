# Extract CLIP features from the dataset

import os, sys
sys.path.insert(0, os.path.abspath("."))

import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPModel
from starter.env_getter import get_env
from src.dataset import Web2MImageDataset
import numpy as np
from tqdm import tqdm
PREP_DIR = get_env('PREP')

if __name__ == '__main__':
    # save_mode = 'distributed'
    save_mode = 'unified'
    split = 'train'
    # split = 'test'
    csv_name = split + '_image_dict_v9.csv'
    model_name = "openai/clip-vit-large-patch14" # [257, 1024]
    keep_pooler = False
    model = CLIPModel.from_pretrained(model_name)

    dataset = Web2MImageDataset(csv_name, split, model_name, pad = True)
    train_dataloader = DataLoader(dataset, shuffle=False, batch_size=64, num_workers=16)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    feature_list = []
    pooler_list = []

    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        pixel_values = batch.pop("pixel_values").to(device)
        new_paths = batch.pop("new_path")
        pooler_path = batch.pop("pooler_path")

        with torch.no_grad():
            outputs = model.vision_model(pixel_values=pixel_values, return_dict = True)
            pooler_output = outputs['pooler_output']
            image_features = model.visual_projection(pooler_output)
        if keep_pooler:
            pooler_output = pooler_output.detach().to('cpu')
        image_features = image_features.detach().to('cpu')
        
        if save_mode == 'distributed':
            for i, new_path in enumerate(new_paths):
                # clone are needed to truely detach the indexed item
                pooler_feature = pooler_output[i].clone()
                image_feature = image_features[i].clone()
                dir_name = os.path.dirname(new_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                dir_name = os.path.dirname(pooler_path[i])
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name) 

                torch.save(image_feature, new_path)
                torch.save(pooler_feature, pooler_path[i])
        if save_mode == 'unified':
            if keep_pooler:
                pooler_list.append(pooler_output)
            feature_list.append(image_features)

    if save_mode == 'unified':
        if keep_pooler:
            pooler_matrix = torch.cat(pooler_list)
            pooler_path = os.path.join(PREP_DIR, "extracted_image_features", f"{split}_pooler_feature_matrix_second.pt")
            print(pooler_matrix.shape)
            torch.save(pooler_matrix, pooler_path)
        image_matrix = torch.cat(feature_list)
        image_matrix_path = os.path.join(PREP_DIR, "extracted_image_features", f"{split}_image_feature_matrix_second.pt")
        torch.save(image_matrix, image_matrix_path)
        


