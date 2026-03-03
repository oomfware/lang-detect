// #region types

/** float32 weights for a linear model (dense → softmax). */
export type ModelWeights = {
	w: Float32Array;
	b: Float32Array;
	inputSize: number;
	outputSize: number;
};

// #endregion

// #region dequantization

/**
 * dequantizes an int8 array back to float32 using its absmax scale.
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

/**
 * loads int8 quantized weights from a binary buffer and dequantizes to float32.
 *
 * binary format: 2 × f32 scales (wScale, bScale), then weight bytes, then bias bytes.
 *
 * @param bin raw binary weight data
 * @param inputSize number of input features
 * @param outputSize number of output classes
 * @returns dequantized model weights
 */
export const loadWeights = (bin: ArrayBuffer, inputSize: number, outputSize: number): ModelWeights => {
	const view = new DataView(bin);
	const wScale = view.getFloat32(0, true);
	const bScale = view.getFloat32(4, true);

	const wSize = outputSize * inputSize;
	const w = new Int8Array(bin, 8, wSize);
	const b = new Int8Array(bin, 8 + wSize, outputSize);

	return {
		w: dequantize(w, wScale),
		b: dequantize(b, bScale),
		inputSize,
		outputSize,
	};
};

/**
 * loads int6 packed quantized weights from a binary buffer and dequantizes to float32.
 *
 * same header as int8 (2 × f32 scales), but payload is 6-bit packed.
 *
 * @param bin raw binary weight data
 * @param inputSize number of input features
 * @param outputSize number of output classes
 * @returns dequantized model weights
 */
export const loadWeights6 = (bin: ArrayBuffer, inputSize: number, outputSize: number): ModelWeights => {
	const view = new DataView(bin);
	const wScale = view.getFloat32(0, true);
	const bScale = view.getFloat32(4, true);

	const wCount = outputSize * inputSize;
	const wPackedSize = Math.ceil((wCount * 3) / 4);
	const bPackedSize = Math.ceil((outputSize * 3) / 4);

	const wPacked = new Uint8Array(bin, 8, wPackedSize);
	const bPacked = new Uint8Array(bin, 8 + wPackedSize, bPackedSize);

	return {
		w: dequantize(unpack6(wPacked, wCount), wScale),
		b: dequantize(unpack6(bPacked, outputSize), bScale),
		inputSize,
		outputSize,
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
