const { downloadModel, loadModel, hash } = require('@proteus-labs/dinohash');
const path = require('path');
const { getModelUrl } = require('../src/utils');

async function main() {
  const modelName = "ResNet-101-Efficient-B0_scripted" // Specify the model to use
  const modelUrl = getModelUrl(modelName);　
  const modelPath = path.join(__dirname, `./models/${modelName}.onnx`);
  const imagePaths = [path.join(__dirname, 'test.png'), path.join(__dirname, 'test.png')];

  await downloadModel(modelUrl, modelPath);
  const session = await loadModel(modelPath, device='cpu'); // can use 'cuda' for GPU inference if you have the right setup
  const results = await hash(session, imagePaths);

  console.log(results[0]);
}

main();