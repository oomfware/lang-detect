"""
dataset loading for lang-detect training.

provides loaders for Tatoeba, UDHR, and Leipzig Corpora Collection,
with a unified merge function for combining data sources.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# ── constants ──

RESOURCES_PATH = Path(__file__).parent / "resources"
TATOEBA_PATH = RESOURCES_PATH / "tatoeba.csv"
UDHR_PATH = RESOURCES_PATH / "udhr"
LEIPZIG_PATH = RESOURCES_PATH / "leipzig"

DATASET_TRAIN_LENGTH_MIN = 40
DATASET_TRAIN_LIMIT = 9000

NGRAM_TYPES = ["unigrams", "bigrams", "trigrams", "quadgrams"]

# ── text normalization ──

HYPHEN_RE = re.compile(r"-+")
MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _is_letter_mark_or_space(c: str) -> bool:
	r"""match JS regex [^\p{L}\p{M}\s] — keep letters, marks, and whitespace."""
	cat = unicodedata.category(c)
	return cat.startswith("L") or cat.startswith("M") or c.isspace()


def normalize(text: str) -> str:
	"""normalize text for ngram extraction, matching the JS implementation."""
	text = HYPHEN_RE.sub(" ", text)
	text = "".join(c for c in text if _is_letter_mark_or_space(c))
	text = MULTI_SPACE_RE.sub(" ", text)
	text = text.lower().strip()
	return f" {text} "


@dataclass
class NgramData:
	counts: dict[str, int]
	freqs: dict[str, float]


def extract_ngrams(text: str, n: int) -> NgramData:
	"""extract ngram counts and frequencies from normalized text."""
	counts: dict[str, int] = {}
	total = 0
	for i in range(len(text) - n + 1):
		gram = text[i:i + n]
		counts[gram] = counts.get(gram, 0) + 1
		total += 1

	if total == 0:
		return NgramData(counts={}, freqs={})
	freqs = {gram: count / total for gram, count in counts.items()}
	return NgramData(counts=counts, freqs=freqs)


# ── data types ──

@dataclass
class RawDatum:
	lang: str
	sentence: str
	ngrams: dict[str, NgramData]  # {type: NgramData}


def make_datum(sentence: str) -> RawDatum:
	"""create a RawDatum from a raw sentence string."""
	norm = normalize(sentence)
	ngrams = {
		"unigrams": extract_ngrams(norm, 1),
		"bigrams": extract_ngrams(norm, 2),
		"trigrams": extract_ngrams(norm, 3),
		"quadgrams": extract_ngrams(norm, 4),
	}
	return RawDatum(lang="", sentence=sentence, ngrams=ngrams)


# ── Tatoeba ──

def load_tatoeba(langs_set: set[str], limit: int) -> dict[str, list[RawDatum]]:
	"""load sentences from the Tatoeba dataset using reservoir sampling for uniform selection."""
	# reservoir sampling: uniformly sample `limit` sentences per language in a single pass.
	# short sentences (< DATASET_TRAIN_LENGTH_MIN) are collected separately as fallback.
	primary: dict[str, list[RawDatum]] = {lang: [] for lang in langs_set}
	primary_seen: dict[str, int] = {lang: 0 for lang in langs_set}
	fallback: dict[str, list[RawDatum]] = {lang: [] for lang in langs_set}
	fallback_seen: dict[str, int] = {lang: 0 for lang in langs_set}

	rng = random.Random(42)

	with open(TATOEBA_PATH, "r", encoding="utf-8") as f:
		for line in f:
			parts = line.rstrip("\n").split("\t")
			if len(parts) != 3:
				continue
			_, lang, sentence = parts
			if lang not in langs_set:
				continue

			is_long = len(sentence) >= DATASET_TRAIN_LENGTH_MIN

			if is_long:
				n = primary_seen[lang]
				primary_seen[lang] = n + 1
				if n < limit:
					primary[lang].append((sentence, n))
				else:
					j = rng.randint(0, n)
					if j < limit:
						primary[lang][j] = (sentence, n)
			else:
				n = fallback_seen[lang]
				fallback_seen[lang] = n + 1
				if n < limit:
					fallback[lang].append((sentence, n))
				else:
					j = rng.randint(0, n)
					if j < limit:
						fallback[lang][j] = (sentence, n)

	# convert to RawDatum and fill from fallback if needed
	result: dict[str, list[RawDatum]] = {}
	for lang in langs_set:
		data = [make_datum(s) for s, _ in primary[lang]]
		for d in data:
			d.lang = lang
		needed = max(0, limit - len(data))
		if needed > 0:
			fb = [make_datum(s) for s, _ in fallback[lang][:needed]]
			for d in fb:
				d.lang = lang
			data.extend(fb)
		result[lang] = data

	return result


# ── UDHR ──

# curated mapping from UDHR file codes to our language codes.
# only standard/modern variants — excludes dialects (068=Welche), historical orthographies,
# and non-standard scripts (vie_han, tgl_tglg).
UDHR_CODE_TO_LANG: dict[str, str] = {
	"afr": "afr",
	"bel": "bel",
	"ben": "ben",
	"bul": "bul",
	"cat": "cat",
	"ces": "ces",
	# ckb UDHR is Latin-script Kurmanji; we train Arabic-script Sorani
	"cmn_hans": "cmn",
	"dan": "dan",
	"deu_1996": "deu",
	"ell_monotonic": "ell",
	"eng": "eng",
	"eus": "eus",
	"fin": "fin",
	"fra": "fra",
	"hau_NG": "hau",
	"heb": "heb",
	"hin": "hin",
	"hrv": "hrv",
	"hun": "hun",
	"hye": "hye",
	"ind": "ind",
	"isl": "isl",
	"ita": "ita",
	"jpn": "jpn",
	"kat": "kat",
	"kaz": "kaz",
	"kor": "kor",
	"lit": "lit",
	"mar": "mar",
	"mkd": "mkd",
	"nld": "nld",
	"nob": "nob",
	"pes_1": "pes",
	"pol": "pol",
	"por_BR": "por",
	"por_PT": "por",
	"ron_2006": "ron",
	"run": "run",
	"rus": "rus",
	"slk": "slk",
	"spa": "spa",
	"srp_cyrl": "srp",
	"srp_latn": "srp",
	"swe": "swe",
	"tgl": "tgl",
	"tur": "tur",
	"ukr": "ukr",
	"vie": "vie",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def load_udhr(langs_set: set[str]) -> dict[str, list[RawDatum]]:
	"""load paragraphs from the UDHR HTML declarations."""
	decl_dir = UDHR_PATH / "declaration"
	if not decl_dir.exists():
		return {}

	result: dict[str, list[RawDatum]] = {}

	for code, lang in UDHR_CODE_TO_LANG.items():
		if lang not in langs_set:
			continue

		html_file = decl_dir / f"{code}.html"
		if not html_file.exists():
			continue

		content = html_file.read_text(encoding="utf-8")
		for match in re.finditer(r"<p>(.*?)</p>", content, re.DOTALL):
			text = _HTML_TAG_RE.sub("", match.group(1)).strip()
			if len(text) < 10:
				continue
			datum = make_datum(text)
			datum.lang = lang
			result.setdefault(lang, []).append(datum)

	return result


# ── Leipzig Corpora Collection ──

# mapping from our lang codes to Leipzig corpus directory names.
# these are data-poor languages in Tatoeba that benefit from supplementary data.
LEIPZIG_CORPORA: dict[str, str] = {
	"slk": "slk_newscrawl_2016_100K",
	"nob": "nob_newscrawl_2019_100K",
	"aze": "aze_newscrawl_2013_100K",
	"kaz": "kaz_newscrawl_2016_100K",
	"afr": "afr_newscrawl_2013_100K",
	"run": "run_community_2017",
	"hrv": "hrv_newscrawl_2016_100K",
	"est": "est_newscrawl_2017_100K",
	"eus": "eus_newscrawl_2012_100K",
}

LEIPZIG_SUPPLEMENTARY_LIMIT = 5000


def load_leipzig(langs_set: set[str], limit: int = LEIPZIG_SUPPLEMENTARY_LIMIT) -> dict[str, list[RawDatum]]:
	"""load sentences from Leipzig Corpora Collection files.

	sentences are randomly sampled up to `limit` per language to keep
	supplementary data manageable alongside the primary Tatoeba corpus.
	"""
	if not LEIPZIG_PATH.exists():
		return {}

	rng = random.Random(42)
	result: dict[str, list[RawDatum]] = {}

	for lang, corpus_name in LEIPZIG_CORPORA.items():
		if lang not in langs_set:
			continue

		sentences_file = LEIPZIG_PATH / corpus_name / f"{corpus_name}-sentences.txt"
		if not sentences_file.exists():
			continue

		# read sentences (ID\tSentence format)
		sentences: list[str] = []
		with open(sentences_file, "r", encoding="utf-8") as f:
			for line in f:
				parts = line.rstrip("\n").split("\t", 1)
				if len(parts) != 2:
					continue
				sentence = parts[1].strip()
				if len(sentence) >= 10:
					sentences.append(sentence)

		# sample if over limit
		if len(sentences) > limit:
			sentences = rng.sample(sentences, limit)

		data: list[RawDatum] = []
		for s in sentences:
			datum = make_datum(s)
			datum.lang = lang
			data.append(datum)

		if data:
			result[lang] = data

	return result


# ── dataset merge ──

def load_dataset_raw(
	langs: list[str],
	limit: int = DATASET_TRAIN_LIMIT,
) -> dict[str, list[RawDatum]]:
	"""load and merge training data from all sources.

	@param langs list of language codes to load
	@param limit max sentences per language from Tatoeba (reservoir sampling)
	@returns merged dataset by language
	"""
	langs_set = set(langs)

	tatoeba = load_tatoeba(langs_set, limit)
	udhr = load_udhr(langs_set)
	leipzig = load_leipzig(langs_set)

	result: dict[str, list[RawDatum]] = {}
	for lang in langs:
		t_data = tatoeba.get(lang, [])
		u_data = udhr.get(lang, [])
		l_data = leipzig.get(lang, [])
		result[lang] = t_data + u_data + l_data

	return result
