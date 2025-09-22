import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from apgd_attack import APGDAttack
from paper_results.temp_neuralhash import NeuralHash
import time
from torchvision import  transforms
import onnx
import os
import seaborn as sns
import matplotlib.pyplot as plt

preprocess = lambda x: np.array(x.convert("RGB").resize([360, 360])).astype(np.float32) / 255.0

class ImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        return preprocess(image).transpose(2,0,1), self.image_files[idx]

epsilon = 32/255

onnx_model = onnx.load("./hashes/model.onnx")
neuralhash = NeuralHash(onnx_model).cuda()
apgd = APGDAttack(neuralhash, eps=epsilon)

folder = "/mnt/additional-disk/home/singh/benchmarking/diffusion_data"
image_files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
image_files.sort()
image_files = image_files[-100_000:]

dataset = ImageDataset(image_files)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

accs = 0
count = 0

to_pil = transforms.ToPILImage()
hashes = []

for image_tensors, labels in tqdm(dataloader):

    image_tensors = image_tensors.cuda()
    # image_before_attack = image_tensors[0]
    # to_pil(image_before_attack).save(f"image_before_attack.png")

    logits = neuralhash.hash(image_tensors, logits=True, differentiable=False)
    logits_bits = (logits >= 0).cpu().numpy()
    hashes.extend(logits_bits)

    adv_images, _ = apgd.attack_single_run(image_tensors, logits, n_iter=10)

    # image_after_attack = adv_images[0] 
    # to_pil(image_after_attack).save(f"image_after_attack.png")

    adv_logits = neuralhash.hash(adv_images, logits=True, differentiable=False)

    adv_logits_bits = (adv_logits[0]>=0).flatten()
    logits_bits = (logits[0]>=0).flatten()


    diff = (adv_logits_bits != logits_bits).sum()

    acc = ((adv_logits>=0) == (logits>=0)).float().mean(1).cpu().numpy()
    accs += acc.sum().item()
    count += len(acc)
    if count % 4096 == 0:
        print(f"Processed {count} images, current accuracy: {accs / count:.6f}")

print(f"Final accuracy after attack: {accs / count:.6f}")

hashes_array = np.array(hashes)

print(hashes_array.astype(int))
correlation_matrix = np.corrcoef(hashes_array.T)
# Flatten the correlation matrix and keep only non-diagonal values
non_diagonal_values = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
plt.hist(non_diagonal_values, bins=50, density=True)
plt.ylabel('Density')
plt.xlabel('Correlation Coefficient')
plt.plot()
plt.savefig('correlation_histogram.png')
plt.close()

means = hashes_array.mean(axis=0)
plt.hist(means, bins=50, density=True)
plt.ylabel('Density')
plt.xlabel('Mean Value')
plt.plot()
plt.savefig('mean_histogram.png')