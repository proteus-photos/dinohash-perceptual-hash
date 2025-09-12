const downloader = require('./src/downloader');
const model = require('./src/model');
const inference = require('./src/inference');
const utils = require("./src/utils");

module.exports = {
  downloadModel: downloader.downloadModel,
  loadModel: model.loadModel,
  hash: inference.hash,
  getModelUrl: utils.getModelUrl,
  getModels: utils.getModels
};
