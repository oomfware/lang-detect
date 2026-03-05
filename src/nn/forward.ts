// #region types

/** ngram vocabulary lists that define the input vector layout for a group model. */
export type GroupNgrams = {
	unigrams: string[];
	bigrams: string[];
	trigrams: string[];
	quadgrams: string[];
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
		const scale = scales[row];
		const off = row * cols;
		for (let col = 0; col < cols; col++) {
			result[off + col] = data[off + col] / scale;
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

/**
 * reads `count` null-terminated UTF-8 strings starting at `offset` in the buffer.
 *
 * @param bytes raw binary data
 * @param offset byte offset to start reading
 * @param count number of strings to read
 * @returns the strings and the byte offset after the last null terminator
 */
const readStrings = (bytes: Uint8Array, offset: number, count: number): [string[], number] => {
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
 * loads a group model from a v2 binary file containing metadata + quantized weights.
 *
 * binary format:
 *   header (16 bytes): "LD" version(2) quantBits outputSize inputSize ngramCounts[4]
 *   strings: null-terminated UTF-8 (langs then ngrams in type order)
 *   scales: f32le[outputSize] wScales, f32le bScale
 *   data: quantized weights (int8 or packed int6), then bias
 *
 * @param bin raw binary data
 * @returns the loaded model ready for inference
 */
export const loadModel = (bin: ArrayBuffer): ReadyModel => {
	const bytes = new Uint8Array(bin);
	const view = new DataView(bin);

	// header
	const quantBits = bytes[3];
	const outputSize = view.getUint16(4, true);
	const inputSize = view.getUint16(6, true);
	const ngramCounts = [
		view.getUint16(8, true),
		view.getUint16(10, true),
		view.getUint16(12, true),
		view.getUint16(14, true),
	];

	// strings
	let pos = 16;
	const [langs, afterLangs] = readStrings(bytes, pos, outputSize);
	pos = afterLangs;
	const [allNgrams, afterNgrams] = readStrings(bytes, pos, inputSize);
	pos = afterNgrams;

	// split ngrams by type
	let ni = 0;
	const ngrams: GroupNgrams = {
		unigrams: allNgrams.slice(ni, (ni += ngramCounts[0])),
		bigrams: allNgrams.slice(ni, (ni += ngramCounts[1])),
		trigrams: allNgrams.slice(ni, (ni += ngramCounts[2])),
		quadgrams: allNgrams.slice(ni, (ni += ngramCounts[3])),
	};

	// scales (per-row weight scales + single bias scale)
	const wScales = new Float32Array(outputSize);
	for (let i = 0; i < outputSize; i++) {
		wScales[i] = view.getFloat32(pos + i * 4, true);
	}
	pos += outputSize * 4;
	const bScale = view.getFloat32(pos, true);
	pos += 4;

	// quantized data
	const wCount = outputSize * inputSize;
	let w: Float32Array;
	let b: Float32Array;

	if (quantBits === 6) {
		const wPackedSize = Math.ceil((wCount * 3) / 4);
		const bPackedSize = Math.ceil((outputSize * 3) / 4);
		w = dequantizePerRow(unpack6(bytes.subarray(pos, pos + wPackedSize), wCount), wScales, inputSize);
		pos += wPackedSize;
		b = dequantize(unpack6(bytes.subarray(pos, pos + bPackedSize), outputSize), bScale);
	} else {
		const wData = new Int8Array(bin, pos, wCount);
		w = dequantizePerRow(wData, wScales, inputSize);
		pos += wCount;
		const bData = new Int8Array(bin, pos, outputSize);
		b = dequantize(bData, bScale);
	}

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
