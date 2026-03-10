const HYPHEN_RE = /-+/g;
const NON_LETTER_RE = /[^\p{L}\p{M}\s]/gu;
const MULTI_SPACE_RE = /\s{2,}/g;

/**
 * normalizes text for ngram extraction: lowercases, strips non-letter/non-mark
 * characters, collapses whitespace, and pads with spaces.
 *
 * @param text raw input text
 * @returns normalized text padded with leading/trailing spaces
 */
export const normalize = (text: string): string => {
	return ` ${text.normalize('NFC').replace(HYPHEN_RE, ' ').replace(NON_LETTER_RE, '').replace(MULTI_SPACE_RE, ' ').toLowerCase().trim()} `;
};

/**
 * extracts ngram frequencies from a string.
 *
 * @param text normalized text (from {@link normalize})
 * @param length ngram length (1 for unigrams, 2 for bigrams, etc.)
 * @returns map of ngram string to its relative frequency (count / total)
 */
export const extractNgrams = (text: string, length: number): Record<string, number> => {
	const ngrams: Record<string, number> = {};
	let total = 0;

	for (let i = 0, l = text.length - length; i <= l; i++) {
		const value = text.slice(i, i + length);
		ngrams[value] = (ngrams[value] || 0) + 1;
		total++;
	}

	for (const value in ngrams) {
		ngrams[value] /= total;
	}

	return ngrams;
};
