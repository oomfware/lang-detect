import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

/**
 * loads binary data from a file URL using node:fs.
 *
 * @param url file URL to load
 * @returns the file contents as an ArrayBuffer
 */
export const loadBinary = async (url: URL): Promise<ArrayBuffer> => {
	const buffer = readFileSync(fileURLToPath(url));
	return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
};
