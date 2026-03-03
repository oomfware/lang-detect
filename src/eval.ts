/**
 * evaluate detection accuracy against the UDHR dataset.
 *
 * usage:
 *   node --conditions source src/eval.ts [--lite] [--lande]
 */

import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';

import { create } from './nn/detect.ts';

const { values: args } = parseArgs({
	options: {
		lite: { type: 'boolean', default: false },
		lande: { type: 'boolean', default: false },
	},
});

const variant = args.lite ? 'lite' : 'standard';
const quantBits = args.lite ? 6 : 8;

const weightsDir = path.resolve(import.meta.dirname!, '..', 'weights', variant);
const { initialize, detect } = create(
	{
		cyrillic: {
			weights: new URL(`file://${path.join(weightsDir, 'cyrillic.bin')}`),
			meta: new URL(`file://${path.join(weightsDir, 'cyrillic.json')}`),
		},
		arabic: {
			weights: new URL(`file://${path.join(weightsDir, 'arabic.bin')}`),
			meta: new URL(`file://${path.join(weightsDir, 'arabic.json')}`),
		},
		devanagari: {
			weights: new URL(`file://${path.join(weightsDir, 'devanagari.bin')}`),
			meta: new URL(`file://${path.join(weightsDir, 'devanagari.json')}`),
		},
		latin: {
			weights: new URL(`file://${path.join(weightsDir, 'latin.bin')}`),
			meta: new URL(`file://${path.join(weightsDir, 'latin.json')}`),
		},
	},
	quantBits,
);

// ── UDHR code → ISO 639-3 mapping ──

const UDHR_CODE_TO_LANG: Record<string, string> = {
	afr: 'afr',
	bel: 'bel',
	ben: 'ben',
	bul: 'bul',
	cat: 'cat',
	ces: 'ces',
	ckb: 'ckb',
	cmn_hans: 'cmn',
	dan: 'dan',
	deu_1996: 'deu',
	ell_monotonic: 'ell',
	eng: 'eng',
	eus: 'eus',
	fin: 'fin',
	fra: 'fra',
	hau_NG: 'hau',
	heb: 'heb',
	hin: 'hin',
	hrv: 'hrv',
	hun: 'hun',
	hye: 'hye',
	ind: 'ind',
	isl: 'isl',
	ita: 'ita',
	jpn: 'jpn',
	kat: 'kat',
	kaz: 'kaz',
	kor: 'kor',
	lit: 'lit',
	mar: 'mar',
	mkd: 'mkd',
	nld: 'nld',
	nob: 'nob',
	pes_1: 'pes',
	pol: 'pol',
	por_BR: 'por',
	por_PT: 'por',
	ron_2006: 'ron',
	run: 'run',
	rus: 'rus',
	slk: 'slk',
	spa: 'spa',
	srp_cyrl: 'srp',
	srp_latn: 'srp',
	swe: 'swe',
	tgl: 'tgl',
	tur: 'tur',
	ukr: 'ukr',
	vie: 'vie',
};

const TAG_RE = /<[^>]+>/g;

// ── load UDHR sentences ──

const declDir = path.resolve(import.meta.dirname!, '..', 'train', 'resources', 'udhr', 'declaration');
const sentences: { lang: string; text: string }[] = [];

for (const [code, lang] of Object.entries(UDHR_CODE_TO_LANG)) {
	const htmlFile = path.join(declDir, `${code}.html`);
	if (!fs.existsSync(htmlFile)) {
		continue;
	}

	const content = fs.readFileSync(htmlFile, 'utf-8');
	for (const match of content.matchAll(/<p>(.*?)<\/p>/gs)) {
		const text = match[1].replace(TAG_RE, '').trim();
		if (text.length < 10) {
			continue;
		}
		sentences.push({ lang, text });
	}
}

// ── helpers ──

type Stats = { pass: number; total: number };

const evaluate = (name: string, detectFn: (text: string) => string | undefined) => {
	const perLang: Record<string, Stats> = {};
	let totalPass = 0;

	for (const { lang, text } of sentences) {
		perLang[lang] ??= { pass: 0, total: 0 };
		perLang[lang].total++;

		if (detectFn(text) === lang) {
			perLang[lang].pass++;
			totalPass++;
		}
	}

	const overallAcc = (totalPass / sentences.length) * 100;

	console.log(`\n=== ${name} ===`);
	console.log(`${sentences.length} sentences, ${Object.keys(perLang).length} languages`);
	console.log(`overall accuracy: ${overallAcc.toFixed(2)}%`);

	const sorted = Object.entries(perLang).sort((a, b) => a[1].pass / a[1].total - b[1].pass / b[1].total);
	for (const [lang, stats] of sorted) {
		const acc = (stats.pass / stats.total) * 100;
		if (acc < 100) {
			console.log(`  ${lang}: ${acc.toFixed(1)}% (${stats.total})`);
		}
	}
};

// ── evaluate ──

await initialize();

evaluate(`UDHR: ${variant} (${quantBits}-bit)`, (text) => {
	const result = detect(text);
	return result[0]?.[0];
});

if (args.lande) {
	// eslint-disable-next-line @typescript-eslint/no-require-imports
	const { default: lande } = await import('lande');
	evaluate('UDHR: lande', (text) => {
		const result = lande(text);
		return result?.[0]?.[0];
	});
}
