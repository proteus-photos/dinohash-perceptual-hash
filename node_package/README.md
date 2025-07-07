# DINOHash

Official implementation of DINOHash, from https://www.arxiv.org/abs/2503.11195, in Node.js

## Installation
`npm install @proteus-labs/dinohash`

## Usage

```
const { downloadModel, loadModel, hash } = require('@proteus-labs/dinohash');
const path = require('path');

async function main() {
  const modelUrl = 'https://huggingface.co/backslashh/DINOHash/resolve/main/dinov2_vits14_reg_96bit_dynamic.onnx';
  // For the 512 bit hash, use dinov2_vitb14_reg_512bit_dynamic.onnx
  
  const modelPath = path.join(__dirname, './models/dinov2_vits14_reg_96bit.onnx');
  const imagePaths = [path.join(__dirname, 'test.png'), path.join(__dirname, 'test.png')];
  
  await downloadModel(modelUrl, modelPath);
  const session = await loadModel(modelPath, device='cpu'); // can use 'cuda' for GPU inference if you have the right setup
  const results = await hash(session, imagePaths);
  console.log(results);
}

main();
```