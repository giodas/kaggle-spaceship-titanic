import * as tf from '@tensorflow/tfjs';

export function createModel(inputDim) {
  const model = tf.sequential();
  const l2 = tf.regularizers.l2({ l2: 1e-4 });
  model.add(tf.layers.dense({ inputShape: [inputDim], units: 128, activation: 'relu', kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu', kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu', kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', kernelRegularizer: l2 }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return model;
}

