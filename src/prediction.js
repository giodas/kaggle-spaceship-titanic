import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

async function main() {
  // Load preprocessing artifacts
  const artifactsPath = '../model_artifacts/preprocessing.json';
  if (!fs.existsSync(artifactsPath)) {
    throw new Error('Missing preprocessing artifacts at ' + artifactsPath);
  }
  const artifacts = JSON.parse(fs.readFileSync(artifactsPath, 'utf-8'));
  const {
    featureNames,
    numericIndices,
    stringIndices,
    numericMeans,
    vocabByFeature,
    oneHotOffsets,
    totalDim
  } = artifacts;

  // Rebuild index maps (token -> position) for string features
  const indexByFeature = {};
  Object.entries(vocabByFeature).forEach(([feat, vocab]) => {
    const map = {};
    vocab.forEach((tok, i) => { map[tok] = i; });
    indexByFeature[feat] = map;
  });

  // Load model
  const model = await tf.loadLayersModel('file://../model_artifacts/model/model.json');
  const modelInputDim = model.inputs[0].shape[1];
  if (modelInputDim !== totalDim) {
    console.warn(`Warning: artifact totalDim (${totalDim}) != model input dim (${modelInputDim}). Will slice/pad.`);
  }

  // Create test dataset (no label column)
  const testDataset = tf.data.csv(
    'file://../data/test.csv',
    { hasHeader: true }
  );

  // Encoding function (mirrors training)
  function encodeRow(xs) {
    const vec = new Array(totalDim).fill(0);

    // Numeric with mean imputation
    numericIndices.forEach((colIdx, pos) => {
      const name = featureNames[colIdx];
      const v = xs[name];
      vec[pos] = (typeof v === 'number' && Number.isFinite(v)) ? v : numericMeans[pos];
    });

    // String one-hot
    stringIndices.forEach(i => {
      const name = featureNames[i];
      if (!oneHotOffsets[name]) return; // Should not happen if artifacts consistent
      const { offset, size } = oneHotOffsets[name];
      let v = xs[name];
      if (v === null || v === undefined || v === '') v = '__MISSING__';
      v = String(v);
      const map = indexByFeature[name];
      const idx = (map && map[v] !== undefined) ? map[v] : (map ? map['__MISSING__'] : -1);
      if (idx >= 0 && idx < size) vec[offset + idx] = 1;
    });

    return vec;
  }

  // Collect rows
  const featureRows = [];
  const passengerIds = [];
  await testDataset.forEachAsync(xs => {
    // xs is the features object
    // PassengerId assumed present among string features
    passengerIds.push(xs.PassengerId ?? '');
    featureRows.push(encodeRow(xs));
  });

  console.log(`Collected ${featureRows.length} test rows.`);

  if (featureRows.length === 0) {
    console.error('No test rows found. Exiting.');
    return;
  }

  // Build tensor
  let featTensor = tf.tensor2d(featureRows, [featureRows.length, totalDim]);

  // Adjust shape if mismatch
  if (modelInputDim !== totalDim) {
    if (totalDim > modelInputDim) {
      featTensor = featTensor.slice([0, 0], [-1, modelInputDim]);
    } else {
      // Pad with zeros
      const padCols = modelInputDim - totalDim;
      const padding = tf.zeros([featureRows.length, padCols]);
      featTensor = tf.concat([featTensor, padding], 1);
      padding.dispose();
    }
  }

  // Predict
  const probs = model.predict(featTensor);
  const probArray = (await probs.array()).map(r => r[0]);
  probs.dispose();
  featTensor.dispose();

  // Threshold
  const predictions = probArray.map(p => p >= 0.5);

  // Write submission CSV
  const outLines = ['PassengerId,Transported'];
  for (let i = 0; i < predictions.length; i++) {
    const pid = passengerIds[i] ?? '';
    outLines.push(`${pid},${predictions[i] ? 'True' : 'False'}`);
  }
  const outPath = '../data/submission.csv';
  fs.writeFileSync(outPath, outLines.join('\n'));
  console.log(`Wrote predictions to ${outPath}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});