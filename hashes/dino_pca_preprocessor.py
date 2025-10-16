import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import torch.nn.functional as F

from torchvision.transforms import functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from transformers import AutoModel
from transformer import Transformer
cudnn.benchmark = True

import webdataset as wds # Import webdataset

BATCH_SIZE = 1024
VERSION ="v3"
num_shards = 120

t = Transformer()
def combined_transform(image):
    transformations = ["erase", "screenshot", "brightness", "text", "contrast", "median"]
    # transformations = []
    random.shuffle(transformations)

    for transform in transformations:
        image = t.transform(image, method=transform)
    return F.resize(image, (224,224))


all_shards_pattern = f"/mnt/unified_dataset/shards/{{00000000..{num_shards-1:08d}}}.tar"
print(f"Loading data from pattern: {all_shards_pattern}")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

crazy_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    combined_transform,
    transforms.ToTensor(),
    normalize
])

dataset = (
    wds.WebDataset(all_shards_pattern, handler=wds.handlers.warn_and_continue, empty_check=False)
    .decode("pil") # Decode the image data into PIL images
    .to_tuple("jpg") # Extract the file with the .jpg extension into a tuple
    .map(lambda x: preprocess(x[0])) # Apply the preprocessing transform to the image
)

crazy_dataset = (
    wds.WebDataset(all_shards_pattern, handler=wds.handlers.warn_and_continue, empty_check=False)
    .decode("pil") # Decode the image data into PIL images
    .to_tuple("jpg") # Extract the file with the .jpg extension into a tuple
    .map(lambda x: crazy_preprocess(x[0])) # Apply the preprocessing transform to the image
)

for BASE_NAME in ["vits16plus", "convnext-tiny", "vits16"]:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, drop_last=False)
    crazy_dataloader = DataLoader(crazy_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, drop_last=False)

    if VERSION == "v2":
        model = torch.hub.load('facebookresearch/dinov2', f"dinov2_{BASE_NAME}")
    else:
        model = AutoModel.from_pretrained(f"facebook/dinov3-{BASE_NAME}-pretrain-lvd1689m")
    model = model.cuda().eval()

    outputs = []
    for images in tqdm(dataloader):
        images = images.cuda()
        
        with torch.no_grad():
            output = model(images)
            if VERSION == "v3":
                output = output.pooler_output
            output = output.cpu()
        outputs.append(output)
        
    outputs = torch.cat(outputs)
    means = outputs.mean(dim=0, keepdim=True)
    outputs -= means

    if outputs.shape[1] > 512:
        pca = IncrementalPCA(n_components=None, whiten=True)
    else:
        pca = PCA(n_components=None, whiten=True)
    outputs = pca.fit_transform(outputs) >= 0 

    crazy_outputs = []
    for images in tqdm(crazy_dataloader):
        images = images.cuda()
        with torch.no_grad():
            output = model(images)
            if VERSION == "v3":
                output = output.pooler_output
            output = output.cpu()
        crazy_outputs.append(output)
        
    crazy_outputs = torch.cat(crazy_outputs)
    crazy_outputs = pca.transform(crazy_outputs) >= 0

    weights = pca.components_

    matches = (outputs == crazy_outputs).sum(0)
    order = np.argsort(-matches)

    weights = weights[order, :]

    means = means.numpy()[:, order]

    np.save(f"./hashes/dino{VERSION}_{BASE_NAME}_means", means)
    np.save(f"./hashes/dino{VERSION}_{BASE_NAME}_PCA", weights)