#!/usr/bin/env bash
# downloads and extracts training datasets (skips existing files)
set -euo pipefail

RESOURCES="$(dirname "$0")/resources"
mkdir -p "$RESOURCES"

# ── Tatoeba sentences ──

if [ -f "$RESOURCES/tatoeba.csv" ]; then
	echo "tatoeba.csv already exists, skipping"
else
	echo "downloading Tatoeba sentences..."
	wget https://downloads.tatoeba.org/exports/sentences.tar.bz2 -O "$RESOURCES/tatoeba.tar.bz2"
	tar -xf "$RESOURCES/tatoeba.tar.bz2"
	mv sentences.csv "$RESOURCES/tatoeba.csv"
	rm "$RESOURCES/tatoeba.tar.bz2"
	echo "done: $RESOURCES/tatoeba.csv"
fi

# ── UDHR (Universal Declaration of Human Rights) ──

if [ -d "$RESOURCES/udhr" ]; then
	echo "udhr/ already exists, skipping"
else
	echo "downloading UDHR from npm..."
	wget -q https://registry.npmjs.org/udhr/-/udhr-6.0.0.tgz -O "$RESOURCES/udhr.tgz"
	mkdir -p "$RESOURCES/udhr"
	tar -xzf "$RESOURCES/udhr.tgz" -C "$RESOURCES/udhr" --strip-components=1
	rm "$RESOURCES/udhr.tgz"
	echo "done: $RESOURCES/udhr/"
fi

# ── Leipzig Corpora Collection ──
# supplementary data for data-poor languages in Tatoeba.
# format: ID\tSentence per line in {name}-sentences.txt

LEIPZIG_DIR="$RESOURCES/leipzig"
LEIPZIG_BASE="https://downloads.wortschatz-leipzig.de/corpora"
mkdir -p "$LEIPZIG_DIR"

download_leipzig() {
	local name="$1"
	if [ -d "$LEIPZIG_DIR/$name" ]; then
		echo "  $name already exists, skipping"
		return
	fi
	echo "  downloading $name..."
	wget -q "$LEIPZIG_BASE/$name.tar.gz" -O "$LEIPZIG_DIR/$name.tar.gz"
	# some tarballs have a top-level directory, others extract flat.
	# always extract into a temp dir, then move the sentences file into place.
	local tmp="$LEIPZIG_DIR/_tmp_$name"
	mkdir -p "$tmp"
	tar -xzf "$LEIPZIG_DIR/$name.tar.gz" -C "$tmp"
	mkdir -p "$LEIPZIG_DIR/$name"
	# handle both: files in subdir or flat
	if [ -f "$tmp/$name/$name-sentences.txt" ]; then
		mv "$tmp/$name/$name-sentences.txt" "$LEIPZIG_DIR/$name/"
	elif [ -f "$tmp/$name-sentences.txt" ]; then
		mv "$tmp/$name-sentences.txt" "$LEIPZIG_DIR/$name/"
	fi
	rm -rf "$tmp" "$LEIPZIG_DIR/$name.tar.gz"
}

echo "downloading Leipzig corpora for data-poor languages..."
download_leipzig "slk_newscrawl_2016_100K"
download_leipzig "nob_newscrawl_2019_100K"
download_leipzig "aze_newscrawl_2013_100K"
download_leipzig "kaz_newscrawl_2016_100K"
download_leipzig "afr_newscrawl_2013_100K"
download_leipzig "run_community_2017"
download_leipzig "hrv_newscrawl_2016_100K"
download_leipzig "est_newscrawl_2017_100K"
download_leipzig "eus_newscrawl_2012_100K"
echo "done: $LEIPZIG_DIR/"
