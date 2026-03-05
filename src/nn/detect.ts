import { loadBinary } from '#load';

import { forward, loadModel, type GroupNgrams, type ReadyModel } from './forward.ts';
import { normalize, extractNgrams } from './normalize.ts';

// #region types

/** a single detection result: ISO 639-3 language code and its probability. */
export type Detection = [lang: string, probability: number];

/** returned by {@link create} — call initialize() once, then detect() synchronously. */
type Detector = {
	initialize: () => Promise<void>;
	detect: (text: string) => Detection[];
};

// #endregion

// #region script classification

/** script family identifiers for character classification. */
type ScriptFamily =
	| 'korean'
	| 'georgian'
	| 'armenian'
	| 'bengali'
	| 'greek'
	| 'hebrew'
	| 'cjk_kana'
	| 'cjk_han'
	| 'cyrillic'
	| 'arabic'
	| 'devanagari'
	| 'latin';

/**
 * classifies a character's Unicode codepoint into a script family.
 *
 * @param cp the codepoint to classify
 * @returns the script family, or `null` if not recognized
 */
const classifyCodepoint = (cp: number): ScriptFamily | null => {
	// unique scripts
	if ((cp >= 0xac00 && cp <= 0xd7af) || (cp >= 0x1100 && cp <= 0x11ff)) {
		return 'korean';
	}
	if ((cp >= 0x10a0 && cp <= 0x10ff) || (cp >= 0x2d00 && cp <= 0x2d2f)) {
		return 'georgian';
	}
	if (cp >= 0x0530 && cp <= 0x058f) {
		return 'armenian';
	}
	if (cp >= 0x0980 && cp <= 0x09ff) {
		return 'bengali';
	}
	if ((cp >= 0x0370 && cp <= 0x03ff) || (cp >= 0x1f00 && cp <= 0x1fff)) {
		return 'greek';
	}
	if (cp >= 0x0590 && cp <= 0x05ff) {
		return 'hebrew';
	}

	// CJK
	if ((cp >= 0x3040 && cp <= 0x309f) || (cp >= 0x30a0 && cp <= 0x30ff)) {
		return 'cjk_kana';
	}
	if ((cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf)) {
		return 'cjk_han';
	}

	// NN groups
	if (cp >= 0x0400 && cp <= 0x04ff) {
		return 'cyrillic';
	}
	if ((cp >= 0x0600 && cp <= 0x06ff) || (cp >= 0x0750 && cp <= 0x077f)) {
		return 'arabic';
	}
	if (cp >= 0x0900 && cp <= 0x097f) {
		return 'devanagari';
	}
	if ((cp >= 0x0041 && cp <= 0x005a) || (cp >= 0x0061 && cp <= 0x007a) || (cp >= 0x00c0 && cp <= 0x024f)) {
		return 'latin';
	}

	return null;
};

/** maps unique script families to their ISO 639-3 language code. */
const UNIQUE_SCRIPT_MAP: Partial<Record<ScriptFamily, string>> = {
	korean: 'kor',
	georgian: 'kat',
	armenian: 'hye',
	bengali: 'ben',
	greek: 'ell',
	hebrew: 'heb',
};

/** maps script families to NN group names. */
const SCRIPT_TO_GROUP: Partial<Record<ScriptFamily, string>> = {
	cyrillic: 'cyrillic',
	arabic: 'arabic',
	devanagari: 'devanagari',
	latin: 'latin',
};

// #endregion

// #region inference helpers

/**
 * builds the input feature vector for a group model from normalized text.
 *
 * @param text normalized text
 * @param ngrams the group's ngram vocabulary
 * @returns float32 input vector matching the model's expected layout
 */
const buildInput = (text: string, ngrams: GroupNgrams): Float32Array => {
	const unigrams = extractNgrams(text, 1);
	const bigrams = extractNgrams(text, 2);
	const trigrams = extractNgrams(text, 3);
	const quadgrams = extractNgrams(text, 4);

	const values = [
		...ngrams.unigrams.map((v) => unigrams[v] || 0),
		...ngrams.bigrams.map((v) => bigrams[v] || 0),
		...ngrams.trigrams.map((v) => trigrams[v] || 0),
		...ngrams.quadgrams.map((v) => quadgrams[v] || 0),
	];

	return new Float32Array(values);
};

// #endregion

// #region detection

/**
 * creates a detector for a specific weight variant.
 *
 * call initialize() once to load and dequantize weights, then
 * call detect() synchronously for each input text.
 *
 * @param sources record of group names to their binary file URLs
 * @returns detector with initialize() and detect() methods
 */
export const create = (sources: Record<string, URL>): Detector => {
	let models: Record<string, ReadyModel> | null = null;

	const initialize = async () => {
		const entries = Object.entries(sources);
		const loaded = await Promise.all(entries.map(([, url]) => loadBinary(url).then(loadModel)));

		models = {};
		for (let i = 0; i < entries.length; i++) {
			models[entries[i][0]] = loaded[i];
		}
	};

	const detect = (text: string): Detection[] => {
		if (!models) {
			throw new Error(`call initialize() first`);
		}

		// classify characters by script family
		const scriptCounts = new Map<ScriptFamily, number>();
		let totalClassified = 0;

		for (let i = 0; i < text.length; i++) {
			const cp = text.codePointAt(i)!;
			// skip surrogates for astral characters
			if (cp > 0xffff) {
				i++;
			}
			const family = classifyCodepoint(cp);
			if (family) {
				scriptCounts.set(family, (scriptCounts.get(family) || 0) + 1);
				totalClassified++;
			}
		}

		// no classified characters — fallback to latin
		if (totalClassified === 0) {
			return detectGroup(text, 'latin', models);
		}

		const results: Detection[] = [];

		for (const [family, count] of scriptCounts) {
			const proportion = count / totalClassified;

			// unique script languages — use proportion directly as probability
			const uniqueLang = UNIQUE_SCRIPT_MAP[family];
			if (uniqueLang) {
				results.push([uniqueLang, proportion]);
				continue;
			}

			// CJK — kana implies Japanese, Han-only implies Chinese
			if (family === 'cjk_kana') {
				results.push(['jpn', proportion]);
				continue;
			}
			if (family === 'cjk_han') {
				// only count as Chinese if no kana detected (otherwise Han is part of Japanese)
				if (!scriptCounts.has('cjk_kana')) {
					results.push(['cmn', proportion]);
				}
				continue;
			}

			// NN group — run model and scale by proportion
			const groupName = SCRIPT_TO_GROUP[family];
			if (groupName && models[groupName]) {
				const groupResults = detectGroup(text, groupName, models, proportion);
				results.push(...groupResults);
			}
		}

		// if nothing was produced (shouldn't happen, but safety), fallback to latin
		if (results.length === 0) {
			return detectGroup(text, 'latin', models);
		}

		results.sort((a, b) => b[1] - a[1]);
		return results;
	};

	return { initialize, detect };
};

/**
 * runs a group's model on the input text and returns detections scaled by proportion.
 *
 * @param text raw input text
 * @param groupName key into the loaded models
 * @param models loaded model records
 * @param proportion script proportion to scale probabilities by
 * @returns detections for this group
 */
const detectGroup = (
	text: string,
	groupName: string,
	models: Record<string, ReadyModel>,
	proportion = 1,
): Detection[] => {
	const model = models[groupName];
	if (!model) {
		throw new Error(`weights not loaded for group '${groupName}'`);
	}

	const normalized = normalize(text);
	const input = buildInput(normalized, model.meta.ngrams);
	const output = forward(input, model.weights);

	const results: Detection[] = model.meta.langs.map((lang, i) => [lang, output[i] * proportion]);
	return results;
};

// #endregion
