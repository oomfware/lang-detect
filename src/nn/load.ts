/**
 * loads binary data from a URL via fetch.
 *
 * @param url URL to fetch
 * @returns the response body as an ArrayBuffer
 */
export const loadBinary = async (url: URL): Promise<ArrayBuffer> => {
	const response = await fetch(url);
	return response.arrayBuffer();
};

/**
 * loads and parses JSON from a URL via fetch.
 *
 * @param url URL to fetch
 * @returns the parsed JSON value
 */
export const loadJson = async (url: URL): Promise<unknown> => {
	const response = await fetch(url);
	return response.json();
};
