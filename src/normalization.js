import * as tf from '@tensorflow/tfjs-node';

/**
 * Compute per-column mean & std ignoring NaNs. All-NaN columns -> mean=0, std=1.
 * @param {tf.Tensor2D} data
 * @returns {{dataMean: tf.Tensor1D, dataStd: tf.Tensor1D, validMask: tf.Tensor1D}}
 */
export function determineMeanAndStddev(data) {
  if (!(data instanceof tf.Tensor)) throw new Error('data must be Tensor');
  if (data.rank !== 2) throw new Error('expected rank-2 tensor');
  return tf.tidy(() => {
    const x = data.dtype === 'float32' ? data : data.toFloat();
    const notNaN = x.isNaN().logicalNot();                // true where value present
    const counts = notNaN.cast('float32').sum(0);         // count per column
    const hasAny = counts.greater(0);                     // bool mask per column
    const cleaned = x.where(notNaN, tf.zerosLike(x));     // replace NaNs with 0 for sums
    const safeCounts = counts.maximum(1);
    let mean = cleaned.sum(0).div(safeCounts);
    const sumSq = cleaned.square().sum(0);
    let variance = sumSq.div(safeCounts).sub(mean.square()).maximum(0);
    let std = variance.sqrt().add(1e-8);                  // avoid divide-by-zero
    // For columns with no valid numbers
    mean = mean.where(hasAny, tf.zerosLike(mean));
    std = std.where(hasAny, tf.onesLike(std));
    return { dataMean: mean, dataStd: std, validMask: hasAny };
  });
}

/**
 * Median per column ignoring NaNs (falls back to mean=0 if all NaN).
 * For simplicity converts column data to JS arrays (okay for moderate feature counts).
 * @param {tf.Tensor2D} data
 * @returns {tf.Tensor1D}
 */
export async function determineMedian(data) {
  if (!(data instanceof tf.Tensor) || data.rank !== 2) throw new Error('determineMedian expects 2D tensor');
  const rows = await data.array(); // [[...]]
  const nCols = rows[0].length;
  const med = new Array(nCols).fill(0);
  for (let c = 0; c < nCols; c++) {
    const colVals = [];
    for (let r = 0; r < rows.length; r++) {
      const v = rows[r][c];
      if (Number.isFinite(v)) colVals.push(v);
    }
    if (colVals.length === 0) { med[c] = 0; continue; }
    colVals.sort((a,b)=>a-b);
    const m = Math.floor(colVals.length / 2);
    med[c] = colVals.length % 2 ? colVals[m] : (colVals[m-1]+colVals[m]) / 2;
  }
  return tf.tensor1d(med, 'float32');
}

/**
 * Replace NaNs with provided per-column fill values.
 * @param {tf.Tensor2D} data
 * @param {tf.Tensor1D} fillValues
 * @returns {tf.Tensor2D}
 */
export function imputeNaN(data, fillValues) {
  return tf.tidy(() => {
    const nanMask = data.isNaN();
    const fillRow = fillValues.reshape([1, -1]).tile([data.shape[0], 1]);
    return data.where(nanMask.logicalNot(), fillRow);
  });
}

/**
 * Normalize numeric columns (those with validMask true). Others (all-NaN) pass through unchanged.
 * @param {tf.Tensor2D} data
 * @param {tf.Tensor1D} dataMean
 * @param {tf.Tensor1D} dataStd
 * @param {tf.Tensor1D} validMask
 */
export function normalizeTensor(data, dataMean, dataStd, validMask) {
  return tf.tidy(() => {
    const norm = data.sub(dataMean).div(dataStd);
    if (!validMask) return norm;
    const mask2D = validMask.reshape([1,-1]).tile([data.shape[0],1]);
    return tf.where(mask2D, norm, data); // non-valid columns unchanged
  });
}
