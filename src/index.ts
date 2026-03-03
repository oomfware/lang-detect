import { create } from './nn/detect.ts';

export type { Detection } from './nn/detect.ts';

export const { initialize, detect } = create({
	cyrillic: {
		weights: new URL('../weights/standard/cyrillic.bin', import.meta.url),
		meta: new URL('../weights/standard/cyrillic.json', import.meta.url),
	},
	arabic: {
		weights: new URL('../weights/standard/arabic.bin', import.meta.url),
		meta: new URL('../weights/standard/arabic.json', import.meta.url),
	},
	devanagari: {
		weights: new URL('../weights/standard/devanagari.bin', import.meta.url),
		meta: new URL('../weights/standard/devanagari.json', import.meta.url),
	},
	latin: {
		weights: new URL('../weights/standard/latin.bin', import.meta.url),
		meta: new URL('../weights/standard/latin.json', import.meta.url),
	},
});
