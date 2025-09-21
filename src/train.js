import * as tf from '@tensorflow/tfjs-node';
import { createModel } from './model.js';

async function run() {

    // Load and preprocess data
    const makeDataset = () => tf.data.csv(
        'file://../data/train.csv', {
            hasHeader: true,
            columnConfigs: { Transported: { isLabel: true } }
        }
    );

    // Initial dataset (for schema + stats)
    let rawDataset = makeDataset();

    // For debugging, take a look at the first element of the dataset.
    rawDataset.take(1).forEachAsync((d) => {
        console.log(d)
    })

    // Peek first row to infer feature names and types
    let featureNames = [];
    let numericIndices = [];
    let stringIndices = [];
    await rawDataset.take(1).forEachAsync(({ xs }) => {
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
    
    // ---- Compute mean for numeric features (ignore missing / non-finite) ----
    const numCount = numericIndices.length;
    const sums = new Array(numCount).fill(0);
    const counts = new Array(numCount).fill(0);

    await rawDataset.forEachAsync(({ xs }) => {
        numericIndices.forEach((colIdx, pos) => {
            const name = featureNames[colIdx];
            const v = xs[name];
            if (typeof v === 'number' && Number.isFinite(v)) {
                sums[pos] += v;
                counts[pos] += 1;
            }
        });
    });

    const numericMeans = sums.map((s, i) => counts[i] > 0 ? s / counts[i] : 0);
    console.log('Numeric means (imputation values):', numericMeans);

    // Re-create dataset for encoding (previous iterator consumed)
    rawDataset = makeDataset();

    // ---- Build vocabularies for string features (with __MISSING__) ----
    const vocabSets = {};
    stringIndices.forEach(i => vocabSets[featureNames[i]] = new Set());

    await rawDataset.forEachAsync(({ xs }) => {
        stringIndices.forEach(i => {
            const key = featureNames[i];
            let v = xs[key];
            if (v === null || v === undefined || v === '') v = '__MISSING__';
            v = String(v);
            vocabSets[key].add(v);
        });
    });

    console.log('Vocab sets:', vocabSets);

    /** 
    This segment finalizes categorical encodings. It starts by preparing two dictionaries: vocabByFeature will hold the ordered list (array) 
    of category strings per feature, and indexByFeature will hold a fast lookup object token -> integer index for each feature.

    For every string (categorical) feature index, it derives the feature name, ensures the special 'MISSING' token is present 
    (so any null/empty encountered later has a defined index), then converts that feature’s Set of tokens into an array. 
    Sorting the array imposes a deterministic, reproducible ordering (important so model weights align across runs and during inference). 
    The sorted array is stored in vocabByFeature.

    Next it builds a plain object map from token to its position (the forEach over vocab with (tok, idx)). This map allows O(1) conversion 
    of raw string values to their integer index during row-to-tensor transformation (e.g., for one‑hot or embedding lookup). 
    That map is stored in indexByFeature keyed by the feature name. The result is a pair of parallel structures: one for iterating (vocab arrays) 
    and one for constant‑time lookup (index maps). A subtle point: relying on lexicographic sort means numeric-looking strings (e.g., '10', '2') 
    will sort as strings; if numeric order is desired, a custom comparator would be needed.
    */
    const vocabByFeature = {};
    const indexByFeature = {};
    stringIndices.forEach(i => {
        const key = featureNames[i];
        if (!vocabSets[key].has('__MISSING__')) vocabSets[key].add('__MISSING__');
        const vocab = Array.from(vocabSets[key]).sort();
        vocabByFeature[key] = vocab;
        const map = {};
        vocab.forEach((tok, idx) => map[tok] = idx);
        indexByFeature[key] = map;
    });

    console.log('Vocab by feature:', vocabByFeature);
    console.log('Index by feature:', indexByFeature);

    /**
    This block precomputes where each categorical feature’s one‑hot segment will live inside a single flattened feature vector. 
    It starts totalDim at the count of numeric features (these will occupy the first slots). For every string feature index it 
    looks up the feature name, then records in oneHotOffsets[key] an object containing: offset (the starting index in the final 
    vector where that feature’s one‑hot slice begins) and size (the vocabulary length, i.e., the width of its one‑hot segment). 
    After storing that, it increments totalDim by the vocabulary size so the next categorical feature’s offset lines up immediately 
    after the previous segment.

    The result is a deterministic, contiguous layout: [ all numeric scalars | one‑hot(feature A) | one‑hot(feature B) | ... ]. 
    Logging the final totalDim confirms the overall input dimensionality (important for creating the tf.input layer or shaping tensors). 
    This approach avoids concatenating multiple small tensors repeatedly; instead you can allocate a single Float32Array (or tensor) 
    of length totalDim and fill numeric values first, then set exactly one index to 1 within each categorical segment using the stored 
    offset + categoryIndex. A potential pitfall is large cardinality features: their vocab length directly inflates totalDim, 
    which can increase memory and slow training; embeddings or hashing would be alternatives if that occurs.
     */
    const oneHotOffsets = {};
    let totalDim = numericIndices.length;
    stringIndices.forEach(i => {
        const name = featureNames[i];
        oneHotOffsets[name] = { offset: totalDim, size: vocabByFeature[name].length };
        totalDim += vocabByFeature[name].length;
    });
    console.log('Total feature vector length (numeric + one-hot):', totalDim);
    console.log('One-hot offsets by feature:', oneHotOffsets);

    function encodeLabel(ys) {
        const v = ys.Transported;
        return (v === true || v === 'True' || v === 'true' || v === 1) ? 1 : 0;
    }

    // Fresh dataset again (third pass) for final encoded batches
    rawDataset = makeDataset();

    // ---- Map rows: mean-impute numeric, one-hot encode strings ----
    const BATCH_SIZE = 32;
    const encodedDataset = rawDataset.map(({ xs, ys }) => {
        const vec = new Array(totalDim).fill(0);

        // Numeric with mean imputation
        numericIndices.forEach((colIdx, pos) => {
            const name = featureNames[colIdx];
            const v = xs[name];
            vec[pos] = (typeof v === 'number' && Number.isFinite(v)) ? v : numericMeans[pos];
        });

        // String one-hot
        /**
        This loop one‑hot encodes every categorical (string) feature into the preallocated flat feature vector vec. 
        For each string feature index it gets the feature’s name, then retrieves that feature’s reserved offset 
        (start position in the combined vector) and its one‑hot segment length (size). It reads the raw value from the 
        current row xs and normalizes missing cases (null, undefined, empty string) to the sentinel '__MISSING__' so they map 
        to a stable bucket.
        The value is coerced to a string to ensure consistent key usage in the lookup map. The code then looks up 
        the integer category index via indexByFeature[name]; if the exact token is absent it falls back to the '__MISSING__' index.
        A safety check ensures the resulting index lies within the segment bounds, and if so it sets exactly one 
        position (offset + idx) to 1, leaving the rest of that segment at 0. This yields a sparse (in content, dense in storage) 
        one‑hot slice per categorical feature placed contiguously after the numeric portion of the feature vector. 
        Potential edge cases: unseen tokens at inference will still resolve via the fallback, and very large vocabularies 
        expand vec length proportionally.
         */
        stringIndices.forEach(i => {
            const name = featureNames[i];
            const { offset, size } = oneHotOffsets[name];
            let v = xs[name];
            if (v === null || v === undefined || v === '') v = '__MISSING__';
            v = String(v);
            const idx = indexByFeature[name][v] !== undefined ? indexByFeature[name][v] : indexByFeature[name]['__MISSING__'];
            if (idx >= 0 && idx < size) vec[offset + idx] = 1;
        });

        return { xs: vec, ys: [encodeLabel(ys)] };
    }).batch(BATCH_SIZE).prefetch(1);

    await encodedDataset.take(1).forEachAsync(b => {
        console.log('Sample batch features shape:', b.xs.shape);
        console.log('Sample batch labels shape:', b.ys.shape);
        console.log('Sample batch features:', b.xs.arraySync());
        console.log('Sample batch labels:', b.ys.arraySync());
    });

    // Create and train model
    const model = createModel(totalDim);
    const EPOCHS = 20;
    console.log('Starting training...');
    await model.fitDataset(encodedDataset, {
        epochs: EPOCHS,
        verbose: 1,
        callbacks: tf.node.tensorBoard('../logdir', {
            updateFreq: 'batch'
        })
    });
    console.log('Training complete.');

    // Persist preprocessing artifacts (include numericMeans)
    const fs = await import('fs');
    if (!fs.existsSync('../model_artifacts')) fs.mkdirSync('../model_artifacts', { recursive: true });
    const artifacts = {
        featureNames,
        numericIndices,
        stringIndices,
        numericMeans,
        vocabByFeature,
        oneHotOffsets,
        totalDim
    };
    fs.writeFileSync('../model_artifacts/preprocessing.json', JSON.stringify(artifacts, null, 2));
    console.log('Saved preprocessing artifacts to model_artifacts/preprocessing.json');

    // Save model
    await model.save('file://../model_artifacts/model');
    console.log('Model saved to model_artifacts/model');
}

run();