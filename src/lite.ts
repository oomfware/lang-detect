import { create } from './nn/detect.ts';

export type { Detection } from './nn/detect.ts';

export const { initialize, detect } = create(
	{
		cyrillic: {
			weights: new URL('../weights/lite/cyrillic.bin', import.meta.url),
			meta: new URL('../weights/lite/cyrillic.json', import.meta.url),
		},
		arabic: {
			weights: new URL('../weights/lite/arabic.bin', import.meta.url),
			meta: new URL('../weights/lite/arabic.json', import.meta.url),
		},
		devanagari: {
			weights: new URL('../weights/lite/devanagari.bin', import.meta.url),
			meta: new URL('../weights/lite/devanagari.json', import.meta.url),
		},
		latin: {
			weights: new URL('../weights/lite/latin.bin', import.meta.url),
			meta: new URL('../weights/lite/latin.json', import.meta.url),
		},
	},
	6,
);
