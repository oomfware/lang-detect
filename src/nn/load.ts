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
