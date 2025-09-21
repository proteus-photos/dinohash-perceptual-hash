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

preprocess = lambda x: np.array(x.convert("RGB").resize([360, 360])).astype(np.float32) / 255.0

class ImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        return preprocess(image).transpose(2,0,1), self.image_files[idx]


# Just loading one image for testing
image_files = ["./test.png"]

dataset = ImageDataset(image_files)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

epsilon = 64

onnx_model = onnx.load("./hashes/model.onnx")
neuralhash = NeuralHash(onnx_model)

apgd = APGDAttack(neuralhash, eps=epsilon, device="cpu")

accs = 0
count = 0

to_pil = transforms.ToPILImage()

for image_tensors, labels in tqdm(dataloader):
    # Save the image before the attack for reference
    image_before_attack = image_tensors[0]
    to_pil(image_before_attack).save(f"image_before_attack.png")

    # Forward pass
    forward_start = time.time()
    logits = neuralhash.forward(image_tensors)
    forward_end = time.time()

    attack_start = time.time()
    adv_images, _ = apgd.attack_single_run(image_tensors, logits, n_iter=10)
    attack_end = time.time()

    # Save the image after the attack for reference
    image_after_attack = adv_images[0] 
    to_pil(image_after_attack).save(f"image_after_attack.png")

    adv_logits = neuralhash.forward(adv_images, c=20, logits=True)

    adv_logits_bits = (adv_logits[0]>=0).flatten()
    logits_bits = (logits[0]>=0).flatten()

    # Print the output hashes for reference
    adv_logits_str = "".join(['1' if b else '0' for b in adv_logits_bits])
    logits_str = "".join(['1' if b else '0' for b in logits_bits])
    print(adv_logits_str)
    print(logits_str)

    diff = (adv_logits_bits != logits_bits).sum()
    print(f"Hamming distance: {diff}")

    acc = ((adv_logits>=0) == (logits>=0)).float().mean(1).cpu().numpy()
    accs += acc.sum().item()
    count += len(acc)
    print(f"Processed {count} images, current accuracy: {accs / count:.6f}")