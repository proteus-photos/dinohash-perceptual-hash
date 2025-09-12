import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from typing import Union, List
import sys
import os

class UniversalHash:
    def __init__(self, pca_dims=None, base_name="dino_mnv2_sslight_traced", device="cuda"):
        self.device = device

        means = np.load(f'./hashes/pca2/{base_name}_means.npy')
        self.means_torch = torch.from_numpy(means).float()

        components = np.load(f'./hashes/pca2/{base_name}_PCA.npy')
        self.components_torch = torch.from_numpy(components).float()

        if pca_dims is None:
            pca_dims = components.shape[1]

        self.pca_dims = pca_dims
        self.base_name = base_name
        path = os.path.join("./hashes/pretrained_models", base_name + ".pt")

        self.model = torch.jit.load(path)
        self.model.eval()

    def hash(
        self,
        image_arrays: Union[np.ndarray, List[Image.Image], torch.Tensor],
        ) -> torch.Tensor:

        if isinstance(image_arrays, np.ndarray):
            image_arrays = torch.from_numpy(image_arrays)
        if isinstance(image_arrays[0], Image.Image):
            image_arrays = torch.stack([preprocess(im) for im in image_arrays])
        if isinstance(image_arrays[0], str):
            image_arrays = torch.stack([preprocess(Image.open(im)) for im in image_arrays])

        with torch.no_grad():
            if (self.device == "cuda"):
                image_arrays = normalize(image_arrays)
            else:
                image_arrays = normalize(image_arrays.cpu())
            
            outs = self.model(image_arrays) - self.means_torch
            
            outs = outs @ self.components_torch

            outs = outs[:, :self.pca_dims] >= 0

        del image_arrays

        return outs


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class Hash:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.cpu()
        self.string = ''.join(str(int(x)) for x in self.tensor)
        self.hex = hex(int(self.string, 2))
        self.array = self.tensor.numpy()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python universal_hash.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path)
    universal_hash = UniversalHash(pca_dims=96, base_name="dino_mnv2_sslight_traced", device="cpu").hash
    hash_tensor = universal_hash([image])[0]

    hash = Hash(hash_tensor)
    print("Perceptual hash:", hash.hex)

