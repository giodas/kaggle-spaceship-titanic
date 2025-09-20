import * as tf from '@tensorflow/tfjs-node';
import { determineMeanAndStddev, determineMedian, imputeNaN, normalizeTensor } from './normalization.js';

async function run() {

    // Load and preprocess data
    const csvDataset = tf.data.csv(
        'file://./data/train.csv', {
            hasHeader: true, 
            columnConfigs: {
                Transported: {
                    isLabel: true
                }
            }
        }
    );

    // For debugging, take a look at the first element of the dataset.
    csvDataset.take(1).forEachAsync((d) => {
        console.log(d)
    })

    // Peek first row to infer feature names and types
    let featureNames = [];
    let numericIndices = [];
    let stringIndices = [];
    await csvDataset.take(1).forEachAsync(({ xs }) => {
        featureNames = Object.keys(xs);
        featureNames.forEach((name, idx) => {
            const v = xs[name];
            if (typeof v === 'number') numericIndices.push(idx);
            else stringIndices.push(idx);
        });
        console.log('Feature names:', featureNames);
        console.log('Numeric indices:', numericIndices);
        console.log('String indices:', stringIndices);
    });

    // Helper to encode boolean/string label to 0/1
    function encodeLabel(ys) {
        const v = ys.Transported;
        if (v === true || v === 'True' || v === 'true' || v === 1) return 1;
        if (v === false || v === 'False' || v === 'false' || v === 0) return 0;
        return NaN; // unexpected / missing
    }

    // Dataset with ONLY numeric features (numbers or NaN placeholders) + numeric label
    const numericDataset = csvDataset.map(({ xs, ys }) => {
        const numArray = numericIndices.map(i => {
            const v = xs[featureNames[i]];
            return (typeof v === 'number') ? v : NaN;
        });
        const label = encodeLabel(ys);
        return { xs: numArray, ys: [label] };
    }).batch(32);

    // Dataset with ONLY string (non-numeric) features + numeric label
    const stringDataset = csvDataset.map(({ xs, ys }) => {
        const strArray = stringIndices.map(i => {
            const v = xs[featureNames[i]];
            if (typeof v === 'string') return v;
            return (v === null || v === undefined) ? '___MISSING___' : String(v);
        });
        const label = encodeLabel(ys);
        return { xs: strArray, ys: [label] };
    }).batch(32);

    // Debug: first numeric batch
    await numericDataset.take(2).forEachAsync(b => {
        console.log('Numeric features batch (numbers/NaN):');
        b.xs.print();
    });

    // Debug: first string batch
    await stringDataset.take(1).forEachAsync(b => {
        console.log('String features batch:');
        b.xs.print(); // dtype 'string'
    });

    // Collect numeric features
    const collected = [];
    await numericDataset.forEachAsync(b => collected.push(b.xs.clone()));
    const allNumeric = tf.concat(collected, 0);

    // Decide imputation strategy: 'mean' or 'median'
    const IMPUTE = 'mean'; // change to 'median' if preferred

    let fillValues;
    if (IMPUTE === 'median') {
        fillValues = await determineMedian(allNumeric);
    } else { // mean
        const { dataMean } = determineMeanAndStddev(allNumeric);
        fillValues = dataMean.clone();
        dataMean.dispose();
    }

    // Impute missing values (NaN -> fillValues)
    const imputedAll = imputeNaN(allNumeric, fillValues);

    // Recompute stats AFTER imputation for normalization
    const { dataMean: finalMean, dataStd: finalStd, validMask } = determineMeanAndStddev(imputedAll);

    console.log('Imputation values:'); fillValues.print();
    console.log('Final mean:'); finalMean.print();
    console.log('Final std:'); finalStd.print();

    // Streaming normalized + imputed dataset
    const imputedNormalizedNumericDataset = numericDataset.map(b => tf.tidy(() => {
        const imputed = imputeNaN(b.xs, fillValues);
        const norm = normalizeTensor(imputed, finalMean, finalStd, validMask);
        return { xs: norm, ys: b.ys };
    }));

    // Build vocabularies for each string feature column (single pass)
    const stringVocabSets = new Array(stringIndices.length).fill(0).map(() => new Set());
    await stringDataset.forEachAsync(batch => batch.xs.array().then(rows => {
        rows.forEach(row => row.forEach((val, c) => {
            const token = (val === null || val === undefined || val === '' ? '___MISSING___' : String(val));
            stringVocabSets[c].add(token);
        }));
    }));
    const stringVocabularies = stringVocabSets.map(s => Array.from(s));
    const stringIndexMaps = stringVocabularies.map(vocab => { const m = new Map(); vocab.forEach((tok,i)=>m.set(tok,i)); return m; });
    console.log('String vocab sizes:', stringVocabularies.map(v => v.length));

    function encodeStringBatch(strTensor) {
        return tf.tidy(() => {
            const arr = strTensor.arraySync(); // [[str,...]]
            const encoded = arr.map(row => row.map((val, c) => {
                const token = (val === null || val === undefined || val === '' ? '___MISSING___' : String(val));
                const idx = stringIndexMaps[c].get(token);
                return idx === undefined ? 0 : idx; // fallback
            }));
            return tf.tensor2d(encoded, [arr.length, stringIndexMaps.length], 'float32');
        });
    }

    // Re-create stringDataset (second pass) to align with numeric batches for merging (with numeric labels)
    const freshStringDataset = csvDataset.map(({ xs, ys }) => {
        const strArray = stringIndices.map(i => {
            const v = xs[featureNames[i]];
            if (typeof v === 'string') return v;
            return (v === null || v === undefined) ? '___MISSING___' : String(v);
        });
        const label = encodeLabel(ys);
        return { xs: strArray, ys: [label] };
    }).batch(32);

    // Merge numeric + encoded string (concatenate feature axis)
    const mergedDataset = tf.data.zip({ num: imputedNormalizedNumericDataset, str: freshStringDataset }).map(({ num, str }) => {
        return tf.tidy(() => {
            const encStr = encodeStringBatch(str.xs);
            const mergedXs = num.xs.concat(encStr, 1);
            return { xs: mergedXs, ys: num.ys }; // labels identical
        });
    });

    // Preview merged batch
    await mergedDataset.take(1).forEachAsync(b => {
        console.log('Merged feature batch (numeric + encoded string):');
        b.xs.print();
        console.log('Labels:');
        b.ys.print();
    });

    // Cleanup (safe because merged dataset tensors are created lazily per iteration)
    collected.forEach(t => t.dispose());
    allNumeric.dispose();
    imputedAll.dispose();
    fillValues.dispose();
    finalMean.dispose();
    finalStd.dispose();
    validMask.dispose();
}

run();