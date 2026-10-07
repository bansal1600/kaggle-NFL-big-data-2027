#!/usr/bin/env bash
# Download the competition files into data/raw/.
# Needs a Kaggle API token in KAGGLE_API_TOKEN (kaggle.com -> Settings -> API -> Create New Token)
# and the competition rules accepted on the Kaggle website.
set -euo pipefail

COMP=nfl-big-data-bowl-2027
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN first}"

mkdir -p "$DEST"
curl -fSL -H "Authorization: Bearer $KAGGLE_API_TOKEN" \
  -o "$DEST/$COMP.zip" "https://www.kaggle.com/api/v1/competitions/data/download-all/$COMP"
unzip -o -q "$DEST/$COMP.zip" -d "$DEST"
rm "$DEST/$COMP.zip"
# The archive nests files in a folder named after the competition; flatten it.
if [ -d "$DEST/$COMP" ]; then mv "$DEST/$COMP"/* "$DEST"/ && rmdir "$DEST/$COMP"; fi
ls -lh "$DEST"
