import * as tf from '@tensorflow/tfjs';

/** 
This function builds and returns a compiled TensorFlow.js Sequential model for binary classification. 
The inputDim argument specifies the length of each flattened feature vector; it becomes the inputShape of the first Dense layer.

A small L2 (weight decay) regularizer (1e-4) is reused across all Dense layers to penalize large weights and reduce overfitting. 
The architecture is a stack of three hidden Dense layers with ReLU activations and unit counts 128 → 64 → 32. 
After each hidden layer a Dropout layer (rate 0.2) randomly zeros 20% of activations during training, adding another form of regularization.

The final Dense layer has a single unit with sigmoid activation, producing a probability (0–1) suitable for a binary target. 
The model is compiled with the 'sgd' optimizer (default learning rate unless overridden), binaryCrossentropy loss, and accuracy metric.
*/

export function createModel(inputDim) {
  const model = tf.sequential();
  const l2 = tf.regularizers.l2({ l2: 1e-4 });
  model.add(tf.layers.dense({ inputShape: [inputDim], units: 128, activation: 'relu', kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu', kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu', kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', kernelRegularizer: l2 }));
  model.compile({ optimizer: 'sgd', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return model;
}

