import * as tf from '@tensorflow/tfjs';

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */

const model = tf.sequential();
model.add(tf.layers.dense({
    inputShape: [13],
    units: 50,
    activation: 'sigmoid',
    kernelInitializer: 'leCunNormal'
}));
model.add(tf.layers.dense(
    { units: 50, activation: 'sigmoid', kernelInitializer: 'leCunNormal' }));
model.add(tf.layers.dense({ units: 1 }));

