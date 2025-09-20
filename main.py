import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import gc
import argparse
from scipy.stats import binom
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from apgd_attack import APGDAttack
from transformer import Transformer
from database import Database
from hashes.dinohash import dinohash, preprocess, load_model

class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(dataset_folder, self.image_files[idx])).convert("RGB")
        if self.transform is not None:
            transformed_image = self.transform(image)
            return preprocess(image), preprocess(transformed_image)
        else:
            return preprocess(image)

def combined_transform(image):
    transformations = ["screenshot", transformation, "erase", "text"]

    if args.attack:
        transformations = []

    for transform in transformations:
        image = t.transform(image, method=transform)
    return image

def generate_roc(matches, bits):
    matches = matches * bits
    taus = np.arange(bits+1)
    tpr = [(matches>=tau).mean() for tau in taus]

    fpr = 1 - binom.cdf(taus-1, bits, 0.5)
    
    df = pd.DataFrame({
        "tpr": tpr,
        "fpr": fpr,
        "tau": taus
    })
    
    df.to_csv(f"./results/{hasher.__name__}_{transformation}.csv")

hasher = dinohash

dataset_folder = './diffusion_data'
image_files = [f for f in os.listdir(dataset_folder)]
image_files.sort()
image_files = image_files[1_800_000:]

BATCH_SIZE = 256
# N_IMAGE_RETRIEVAL = 1

parser = argparse.ArgumentParser(description ='Perform retrieval benchmarking.')
parser.add_argument('-r', '--refresh', action='store_true')
parser.add_argument('--checkpoint', dest='checkpoint', type=str, default=None,
                    help='path to checkpoint')
parser.add_argument('--attack', action='store_true')

parser.add_argument('--transform')
args = parser.parse_args()

transformation = args.transform
t = Transformer()

if args.checkpoint is not None:
    load_model(args.checkpoint)

if args.attack:
    apgd = APGDAttack(eps=4/255)
    print("Not applying any transformations since attack is enabled.")

os.makedirs("databases", exist_ok=True)
if hasher.__name__ + ".npy" not in os.listdir("databases") or args.refresh:
    print("Creating database for", hasher.__name__)
    original_hashes = []

    dataset = ImageDataset(image_files, transform=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=11)

    for image_batch in tqdm(dataloader):
        original_hashes.extend(hasher(image_batch).cpu())
        gc.collect()
        
    db = Database(original_hashes, storedir=f"databases/{hasher.__name__}")
else:
    db = Database(None, storedir=f"databases/{hasher.__name__}")

print(f"Computing bit accuracy for {transformation} + {hasher.__name__}...")
modified_hashes = []
dataset = ImageDataset(image_files, transform=combined_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=11)

for images, transformed_images in tqdm(dataloader):
    # copy = transformed_images.detach().clone()
    if args.attack:
        original_logits = hasher(images, logits=True)
    
        transformed_images, _ = apgd.attack_single_run(transformed_images, original_logits, n_iter=50)

        # we technically don't need to do this but
        # it deflects a bit of the attack by rounding off the pixels 
        pil_images = [F.to_pil_image(image) for image in transformed_images]

    modified_hashes_batch = hasher(transformed_images).tolist()
    modified_hashes.extend(modified_hashes_batch)

modified_hashes = np.array(modified_hashes)
bits = modified_hashes.shape[-1]

matches = db.similarity_score(modified_hashes)
inv_matches = db.similarity_score(modified_hashes[::-1])

print(matches.mean(), matches.std())
print(inv_matches.mean(), inv_matches.std())

with open(f"./results/{hasher.__name__}_{transformation}.txt", "w") as f:
    f.write(f"Bit accuracy: {matches.mean()} / {matches.std()}\n")
    f.write(f"Random accuracy: {inv_matches.mean()} / {inv_matches.std()}\n")

generate_roc(matches, bits=bits)

