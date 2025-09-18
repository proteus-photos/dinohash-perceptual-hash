
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from apgd_attack import APGDAttack
from paper_results.temp_neuralhash import NeuralHash
import time
from torchvision import datasets, transforms

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # convert 1 → 3 channels
    transforms.Resize((360, 360)),
    transforms.ToTensor(),
])

data_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)

mnist_subset = Subset(data_set, range(100))  # first 100 samples
dataloader = DataLoader(mnist_subset, batch_size=256, shuffle=False)

print("Dataloader loaded")

epsilon = 8/255

neuralhash = NeuralHash()
apgd = APGDAttack(neuralhash, eps=epsilon, device="cpu")

accs = 0
count = 0
for image_tensors, labels in tqdm(dataloader):
    # logits = dinohash(image_tensors, differentiable=False, logits=True, prod_output=False)
    forward_start = time.time()
    logits = neuralhash.forward(image_tensors)
    forward_end = time.time()
    print(f"Forward time: {forward_end - forward_start}")


    attack_start = time.time()
    adv_images, _ = apgd.attack_single_run(image_tensors, logits, n_iter=10)
    attack_end = time.time()
    print(f"Attack time: {attack_end - attack_start}")

    adv_logits = neuralhash.forward(adv_images)
    print(adv_logits)

    acc = ((adv_logits>=0) == (logits>=0)).float().mean(1).cpu().numpy()
    accs += acc.sum().item()
    count += len(acc)
    print(f"Processed {count} images, current accuracy: {accs / count:.6f}")