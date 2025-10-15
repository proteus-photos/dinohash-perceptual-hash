import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
import webdataset as wds # Import webdataset

num_shards = 150

all_shards_pattern = f"/mnt/unified_dataset/shards/{{00000000..{num_shards-1:08d}}}.tar"
print(f"Loading data from pattern: {all_shards_pattern}")


# --- Preprocessing and Model Setup (Unchanged) ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 1024
model = "dinov2_vits14_reg"

dataset = (
    wds.WebDataset(all_shards_pattern, handler=wds.handlers.warn_and_continue)
    .decode("pil") # Decode the image data into PIL images
    .to_tuple("jpg") # Extract the file with the .jpg extension into a tuple
    .map(lambda x: preprocess(x[0])) # Apply the preprocessing transform to the image
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=11)

# --- The rest of your script remains exactly the same ---
print("Loading DINOv2 model...")
dinov2 = torch.hub.load('facebookresearch/dinov2', model).cuda().eval()

print("Extracting features...")
outputs = []
for images in tqdm(dataloader):
    images = images.cuda()
    with torch.no_grad():
        output = dinov2(images).cpu()
    outputs.append(output)
    
print("Concatenating features...")
outputs = torch.cat(outputs)
means = outputs.mean(dim=0, keepdim=True)
outputs -= means

pca = PCA(n_components=None, whiten=True)
pca.fit(outputs)

weights = pca.components_

# Create directory for hashes if it doesn't exist
os.makedirs("./hashes", exist_ok=True)

np.save(f"./hashes/{model}_means", means.numpy())
np.save(f"./hashes/{model}_PCA", weights)