
const MODEL_URLS = {
  "ResNet-101-Efficient-B0_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-101–Efficient-B0_scripted.onnx",
  "ResNet-101–Efficient-B1_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-101–Efficient-B1_scripted.onnx",
  "ResNet-101–MobileNet-v3_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-101–MobileNet-v3_scripted.onnx",
  "ResNet-101–ResNet-18_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-101–ResNet-18_scripted.onnx",
  "ResNet-101–ResNet-34_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-101–ResNet-34_scripted.onnx",
  "ResNet-152–Efficient-B0_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-152–Efficient-B0_scripted.onnx",
  "ResNet-152–Efficient-B1_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-152–Efficient-B1_scripted.onnx",
  "ResNet-152–MobileNet-v3_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-152–MobileNet-v3_scripted.onnx",
  "ResNet-152–ResNet-18_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-152–ResNet-18_scripted.onnx",
  "ResNet-152–ResNet-34_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-152–ResNet-34_scripted.onnx",
  "ResNet-50*2–Efficient-B0_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50*2–Efficient-B0_scripted.onnx",
  "ResNet-50*2–Efficient-B1_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50*2–Efficient-B1_scripted.onnx",
  "ResNet-50*2–MobileNet-v3_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50*2–MobileNet-v3_scripted.onnx",
  "ResNet-50*2–ResNet-18_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50*2–ResNet-18_scripted.onnx",
  "ResNet-50*2–ResNet-34_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50*2–ResNet-34_scripted.onnx",
  "ResNet-50–Efficient-B0_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50–Efficient-B0_scripted.onnx",
  "ResNet-50–Efficient-B1_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50–Efficient-B1_scripted.onnx",
  "ResNet-50–MobileNet-v3_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50–MobileNet-v3_scripted.onnx",
  "ResNet-50–ResNet-18_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50–ResNet-18_scripted.onnx",
  "ResNet-50–ResNet-34_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-50–ResNet-34_scripted.onnx",
  "dino_mnv2_sslight_traced": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/dino_mnv2_sslight_traced.onnx",
  "dino_vitt16_sslight_traced": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/dino_vitt16_sslight_traced.onnx",
  "moco_mnv2_sslight_traced": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/moco_mnv2_sslight_traced.onnx",
  "swav_mnv2_sslight_traced": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/swav_mnv2_sslight_traced.onnx",
  "ResNet-101–Efficient-B0_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/ResNet-101–Efficient-B0_scripted.onnx",
  "dino_r18_sslight_scripted":"https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/dino_r18_sslight_scripted.onnx",
  "dino_r34_sslight_scripted": "https://huggingface.co/dantehrani/proteus-models/resolve/main/onnx/dino_r34_sslight_scripted.onnx"
}

const getModels = () => {
  return Object.keys(MODEL_URLS)
}

const getModelUrl = (modelName) => {
    if (Object.keys(MODEL_URLS).includes(!modelName)) {
        throw new Error(`Model ${modelName} not found`)
    }

    return MODEL_URLS[modelName]
}


module.exports = {
  getModels,
  getModelUrl
}