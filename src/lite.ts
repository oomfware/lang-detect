import { create } from './nn/detect.ts';

export type { Detection } from './nn/detect.ts';

export const { initialize, detect } = create({
	cyrillic: new URL('../weights/lite/cyrillic.bin', import.meta.url),
	arabic: new URL('../weights/lite/arabic.bin', import.meta.url),
	devanagari: new URL('../weights/lite/devanagari.bin', import.meta.url),
	latin: new URL('../weights/lite/latin.bin', import.meta.url),
});
