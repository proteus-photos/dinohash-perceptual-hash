import torch
import numpy as np
import torch.nn as nn
import os
from huggingface_hub import hf_hub_download, list_repo_files

# Download the model weights from the huggingface repo
def download_model(model_name):
    os.makedirs("./pretrained-models", exist_ok=True)
    print(f"Downloading model {model_name}")

    model_path = hf_hub_download(
        repo_id="dantehrani/proteus-models",
        filename=f"pretrained-models/{model_name}.pt",
        revision="main",
        repo_type="model",
        local_dir="./"
    )

    print(f"Model downloaded to {model_path}")

# Download the model params from the huggingface repo
def download_params(model_name):
    os.makedirs("./params", exist_ok=True)
    print(f"Downloading params for model {model_name}")

    PCA_path = hf_hub_download(
        repo_id="dantehrani/proteus-models",
        filename=f"params/{model_name}_PCA.npy",
        revision="main",
        repo_type="model",
        local_dir="./",
    )

    print(f"PCA downloaded to {PCA_path}")

    means_path = hf_hub_download(
        repo_id="dantehrani/proteus-models",
        filename=f"params/{model_name}_means.npy",
        revision="main",
        repo_type="model",
        local_dir="./",
    )

    print(f"Means downloaded to {means_path}")

# Export a model to ONNX to the onnx folder
def export_model(model_name):
    print(f"Exporting model {model_name} to ONNX")

    download_model(model_name)
    download_params(model_name)

    # Load the model
    model = torch.jit.load(f"./pretrained-models/{model_name}.pt", map_location="cpu").eval()
    
    
    means = np.load(f'./params/{model_name}_means.npy')
    means_torch = torch.from_numpy(means).float()
    
    components = np.load(f'./params/{model_name}_PCA.npy').T
    components = components.T
    components_torch = torch.from_numpy(components).float()

    #! Commented out because this causes the export to fail
    """
    # integrate linear component to mimic PCA
    linear = nn.Linear(components_torch.shape[0], components_torch.shape[1])
    linear.weight.data = nn.parameter.Parameter(components_torch.T)
    linear.bias.data = nn.parameter.Parameter(-means_torch@components_torch)

    model = nn.Sequential(
        model,
        linear
    )
    """

    batch_size = 42  # can be set to anything
    example = torch.rand(batch_size, 3, 224, 224)

    example = torch.rand(1, 3, 224, 224)  # use batch=1 for export; we’ll make it dynamic

    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    torch.onnx.export(
        model,
        (example,),
        f"./onnx/{model_name}.onnx",
        opset_version=17,
        dynamic_axes=dynamic_axes,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True,
    )    

    print(f"Model {model_name} exported to ONNX\n")



if __name__ == "__main__":
    # Get the list of models from the huggingface repo
    files = list_repo_files("dantehrani/proteus-models", repo_type="model")

    model_files = [f for f in files if f.startswith("pretrained-models")]
    model_names = [f.split("/")[1].split(".")[0] for f in model_files]

    # Export the models to ONNX one by one
    for model_name in model_names:
        try:
            export_model(model_name)
        except Exception as e:
            print(f"Error exporting model {model_name}: {e}\n")
