// #region types

/** ngram vocabulary lists that define the input vector layout for a group model. */
export type GroupNgrams = {
	unigrams: string[];
	bigrams: string[];
	trigrams: string[];
	quadgrams: string[];
	pentagrams: string[];
};

/** metadata parsed from a group's binary file header. */
export type GroupMeta = {
	langs: string[];
	ngrams: GroupNgrams;
};

/** float32 weights for a linear model (dense → softmax). */
export type ModelWeights = {
	w: Float32Array;
	b: Float32Array;
	inputSize: number;
	outputSize: number;
};

/** a loaded group model ready for inference. */
export type ReadyModel = {
	meta: GroupMeta;
	weights: ModelWeights;
};

// #endregion

// #region dequantization

/**
 * dequantizes an int8 array back to float32 using a single absmax scale.
 *
 * @param data quantized int8 values
 * @param scale the scale factor used during quantization (scaleMax / absmax)
 * @returns dequantized float32 array
 */
const dequantize = (data: Int8Array, scale: number): Float32Array => {
	const result = new Float32Array(data.length);
	for (let i = 0; i < data.length; i++) {
		result[i] = data[i] / scale;
	}
	return result;
};

/**
 * dequantizes a 2D int8 weight matrix using per-row absmax scales.
 *
 * @param data quantized int8 values (rows × cols, row-major)
 * @param scales per-row scale factors
 * @param cols number of columns per row
 * @returns dequantized float32 array
 */
const dequantizePerRow = (data: Int8Array, scales: Float32Array, cols: number): Float32Array => {
	const result = new Float32Array(data.length);
	for (let row = 0; row < scales.length; row++) {
		// scales[row] is 1/absmax — multiply by absmax in the hot loop
		const invScale = 1 / scales[row];
		const off = row * cols;
		for (let col = 0; col < cols; col++) {
			result[off + col] = data[off + col] * invScale;
		}
	}
	return result;
};

/**
 * unpacks 6-bit packed bytes into signed int8 values.
 *
 * packing scheme: 4 values (6 bits each, unsigned offset by +31) → 3 bytes.
 * byte0 = (u0 << 2) | (u1 >> 4)
 * byte1 = ((u1 & 0x0F) << 4) | (u2 >> 2)
 * byte2 = ((u2 & 0x03) << 6) | u3
 *
 * @param packed packed 6-bit data
 * @param count number of original values
 * @returns signed int8 values in [-31, 31]
 */
const unpack6 = (packed: Uint8Array, count: number): Int8Array => {
	const result = new Int8Array(count);
	let ri = 0;
	let pi = 0;

	// process full groups of 4
	const fullGroups = (count >> 2) << 2;
	while (ri < fullGroups) {
		const b0 = packed[pi];
		const b1 = packed[pi + 1];
		const b2 = packed[pi + 2];
		result[ri] = (b0 >> 2) - 31;
		result[ri + 1] = (((b0 & 0x03) << 4) | (b1 >> 4)) - 31;
		result[ri + 2] = (((b1 & 0x0f) << 2) | (b2 >> 6)) - 31;
		result[ri + 3] = (b2 & 0x3f) - 31;
		ri += 4;
		pi += 3;
	}

	// remainder (1-3 values)
	const rem = count - fullGroups;
	if (rem >= 1) {
		result[ri] = (packed[pi] >> 2) - 31;
	}
	if (rem >= 2) {
		result[ri + 1] = (((packed[pi] & 0x03) << 4) | (packed[pi + 1] >> 4)) - 31;
	}
	if (rem >= 3) {
		result[ri + 2] = (((packed[pi + 1] & 0x0f) << 2) | (packed[pi + 2] >> 6)) - 31;
	}

	return result;
};

// #endregion

// #region binary loading

const decoder = new TextDecoder();

/** sentinel value in the quantBits header byte that signals product quantization. */
const PQ_QUANT = 0xf4;

/**
 * reads `count` null-terminated UTF-8 strings starting at `offset`.
 *
 * @param bytes raw binary data
 * @param offset byte offset to start reading
 * @param count number of strings to read
 * @returns the strings and the byte offset after the last null terminator
 */
const readNullTermStrings = (bytes: Uint8Array, offset: number, count: number): [string[], number] => {
	const strings: string[] = [];
	let pos = offset;
	for (let i = 0; i < count; i++) {
		let end = pos;
		while (bytes[end] !== 0) {
			end++;
		}
		strings.push(decoder.decode(bytes.subarray(pos, end)));
		pos = end + 1;
	}
	return [strings, pos];
};

/**
 * decodes nibble-packed prefix-shared ngram buckets.
 *
 * for each bucket, entries are stored as `u8 (shared<<4) | suffix_len` followed
 * by the suffix bytes. the "previous bytes" buffer resets at bucket boundaries
 * so downstream bucket-slicing by ngramCounts[] still works.
 *
 * @param bytes raw binary data
 * @param offset byte offset to start reading
 * @param counts per-bucket ngram counts (length 5)
 * @returns the decoded strings (flat, in bucket order) and the new byte offset
 */
const readPrefixNgrams = (bytes: Uint8Array, offset: number, counts: number[]): [string[], number] => {
	const strings: string[] = [];
	let pos = offset;
	// 4-bit shared/suffix nibbles cap each entry at 30 bytes
	const scratch = new Uint8Array(32);
	for (let bucket = 0; bucket < counts.length; bucket++) {
		for (let i = 0; i < counts[bucket]; i++) {
			const header = bytes[pos++];
			const shared = header >> 4;
			const suffixLen = header & 0x0f;
			for (let k = 0; k < suffixLen; k++) {
				scratch[shared + k] = bytes[pos + k];
			}
			pos += suffixLen;
			strings.push(decoder.decode(scratch.subarray(0, shared + suffixLen)));
		}
	}
	return [strings, pos];
};

/**
 * decodes a product-quantized weight matrix into a dense float32 matrix.
 *
 * codebook rows hold normalized centroid values; per-row absmax scales undo
 * the normalization applied at encode time. the encoder pads the input
 * dimension up to a multiple of D, so we materialize into the padded width
 * and then return only the first `inputSize` columns.
 *
 * @param codebook K * D float32 centroids (row-major)
 * @param indices outputSize * subvectorsPerRow uint8 codebook indices
 * @param wScales per-row scales stored as 1/absmax (divisor, matching int6 convention)
 * @param outputSize number of output rows
 * @param inputSize true input dimension (unpadded)
 * @param subvectorsPerRow number of subvectors per row (= paddedIn / D)
 * @param d subvector length
 * @returns dequantized float32 weight matrix (outputSize × inputSize, row-major)
 */
const decodePq = (
	codebook: Float32Array,
	indices: Uint8Array,
	wScales: Float32Array,
	outputSize: number,
	inputSize: number,
	subvectorsPerRow: number,
	d: number,
): Float32Array => {
	const result = new Float32Array(outputSize * inputSize);
	// only the last subvector can cross the padding boundary; the preceding ones
	// always fill a full d-wide stride, so run the guard-free loop for those and
	// handle the tail once per row.
	const fullSubvectors = Math.floor(inputSize / d);
	const tailLen = inputSize - fullSubvectors * d;
	for (let row = 0; row < outputSize; row++) {
		// wScales is stored as 1/absmax (divisor); multiply is cheaper in the hot loop
		const invScale = 1 / wScales[row];
		const outOff = row * inputSize;
		const idxOff = row * subvectorsPerRow;
		for (let s = 0; s < fullSubvectors; s++) {
			const cbOff = indices[idxOff + s] * d;
			const colStart = s * d;
			for (let t = 0; t < d; t++) {
				result[outOff + colStart + t] = codebook[cbOff + t] * invScale;
			}
		}
		if (tailLen > 0) {
			const cbOff = indices[idxOff + fullSubvectors] * d;
			const colStart = fullSubvectors * d;
			for (let t = 0; t < tailLen; t++) {
				result[outOff + colStart + t] = codebook[cbOff + t] * invScale;
			}
		}
	}
	return result;
};

/**
 * loads a group model from a binary file produced by `train/export.py`.
 *
 * the `quantBits` header byte drives both weight decoding and ngram string
 * encoding. product quantization pins ngram columns to the codebook's subvector
 * layout, so those groups use null-terminated strings; int6 groups are free
 * to reorder columns and use prefix-shared string encoding.
 *
 *   header (17 bytes): "LD" u8 quantBits u16le outputSize u16le inputSize u16le[5] ngramCounts
 *   langs: null-terminated UTF-8
 *   ngrams (per quantBits):
 *     - 0xF4 (PQ):  null-terminated UTF-8 in logical order
 *     - 6 (int6):   nibble-packed prefix-shared per bucket
 *   scales: f32le[outputSize] wScales + f32le bScale
 *   weight data (per quantBits):
 *     - 0xF4 (PQ):  u16le K, u8 D, u16le subvectorsPerRow,
 *                   f32le[K*D] codebook, u8[outputSize*subvectorsPerRow] indices
 *     - 6 (int6):   packed int6 weights
 *   bias: packed int6 (always)
 *
 * @param bin raw binary data
 * @returns the loaded model ready for inference
 */
export const loadModel = (bin: ArrayBuffer): ReadyModel => {
	const bytes = new Uint8Array(bin);
	const view = new DataView(bin);

	const quantBits = bytes[2];
	const outputSize = view.getUint16(3, true);
	const inputSize = view.getUint16(5, true);
	const ngramCounts = [
		view.getUint16(7, true),
		view.getUint16(9, true),
		view.getUint16(11, true),
		view.getUint16(13, true),
		view.getUint16(15, true),
	];

	let pos = 17;
	const [langs, afterLangs] = readNullTermStrings(bytes, pos, outputSize);
	pos = afterLangs;
	const [allNgrams, afterNgrams] =
		quantBits === PQ_QUANT
			? readNullTermStrings(bytes, pos, inputSize)
			: readPrefixNgrams(bytes, pos, ngramCounts);
	pos = afterNgrams;

	// split ngrams by type
	let ni = 0;
	const ngrams: GroupNgrams = {
		unigrams: allNgrams.slice(ni, (ni += ngramCounts[0])),
		bigrams: allNgrams.slice(ni, (ni += ngramCounts[1])),
		trigrams: allNgrams.slice(ni, (ni += ngramCounts[2])),
		quadgrams: allNgrams.slice(ni, (ni += ngramCounts[3])),
		pentagrams: allNgrams.slice(ni, (ni += ngramCounts[4])),
	};

	// scales (per-row weight scales + single bias scale)
	const wScales = new Float32Array(outputSize);
	for (let i = 0; i < outputSize; i++) {
		wScales[i] = view.getFloat32(pos + i * 4, true);
	}
	pos += outputSize * 4;
	const bScale = view.getFloat32(pos, true);
	pos += 4;

	// quantized weight data
	const wCount = outputSize * inputSize;
	let w: Float32Array;

	if (quantBits === PQ_QUANT) {
		const k = view.getUint16(pos, true);
		pos += 2;
		const d = bytes[pos];
		pos += 1;
		const subvectorsPerRow = view.getUint16(pos, true);
		pos += 2;

		// codebook is f32le[K*D]; align-safe read via DataView
		const codebook = new Float32Array(k * d);
		for (let i = 0; i < codebook.length; i++) {
			codebook[i] = view.getFloat32(pos + i * 4, true);
		}
		pos += codebook.length * 4;

		const idxCount = outputSize * subvectorsPerRow;
		const indices = new Uint8Array(bin, pos, idxCount);
		pos += idxCount;

		w = decodePq(codebook, indices, wScales, outputSize, inputSize, subvectorsPerRow, d);
	} else {
		const wPackedSize = Math.ceil((wCount * 3) / 4);
		w = dequantizePerRow(unpack6(bytes.subarray(pos, pos + wPackedSize), wCount), wScales, inputSize);
		pos += wPackedSize;
	}

	// bias: packed int6
	const bPackedSize = Math.ceil((outputSize * 3) / 4);
	const b = dequantize(unpack6(bytes.subarray(pos, pos + bPackedSize), outputSize), bScale);

	return {
		meta: { langs, ngrams },
		weights: { w, b, inputSize, outputSize },
	};
};

// #endregion

// #region forward pass

/**
 * applies softmax in-place to an output array.
 *
 * @param output logit array to convert to probabilities
 */
const softmax = (output: Float32Array): void => {
	let max = -Infinity;
	for (let i = 0; i < output.length; i++) {
		if (output[i] > max) {
			max = output[i];
		}
	}
	let expSum = 0;
	for (let i = 0; i < output.length; i++) {
		output[i] = Math.exp(output[i] - max);
		expSum += output[i];
	}
	for (let i = 0; i < output.length; i++) {
		output[i] /= expSum;
	}
};

/**
 * forward pass for a linear model: dense → softmax.
 *
 * @param input input feature vector (ngram frequencies)
 * @param m model weights
 * @returns output probabilities (one per language in the group)
 */
export const forward = (input: Float32Array, m: ModelWeights): Float32Array => {
	const output = new Float32Array(m.outputSize);
	for (let i = 0; i < m.outputSize; i++) {
		let sum = m.b[i];
		const off = i * m.inputSize;
		for (let j = 0; j < m.inputSize; j++) {
			sum += input[j] * m.w[off + j];
		}
		output[i] = sum;
	}

	softmax(output);
	return output;
};

// #endregion
