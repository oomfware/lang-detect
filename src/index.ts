import { create } from './nn/detect.ts';

export type { Detection } from './nn/detect.ts';

export const { initialize, detect } = create({
	cyrillic: new URL('../weights/standard/cyrillic.bin', import.meta.url),
	arabic: new URL('../weights/standard/arabic.bin', import.meta.url),
	devanagari: new URL('../weights/standard/devanagari.bin', import.meta.url),
	latin: new URL('../weights/standard/latin.bin', import.meta.url),
});
