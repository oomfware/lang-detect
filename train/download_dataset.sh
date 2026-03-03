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
