#!/usr/bin/env bash
# Re-publish the Cosmos report to the SAME permanent here.now URL.
#   Live: https://wintry-tulip-tj52.here.now/
# Edit index.html (tracked here) and/or regenerate the asset videos, then run this.
set -e
SLUG=wintry-tulip-tj52
HERE="$(cd "$(dirname "$0")" && pwd)"
ASSETS="$HERE/../data_overlay_industrial/cosmos/report"   # gitignored; holds the .mp4s
SKILL="$HOME/.claude/skills/here-now/scripts/publish.sh"
TMP="$(mktemp -d)"
cp "$HERE/index.html" "$TMP/"
cp "$ASSETS"/*.mp4 "$TMP/" 2>/dev/null || { echo "No videos in $ASSETS — regenerate via the cosmos pipeline first."; exit 1; }
bash "$SKILL" "$TMP" --slug "$SLUG" \
  --title "Cross-Embodiment -> Cosmos: New Backgrounds" \
  --description "Generating new backgrounds for the YAM robot feed with Cosmos Transfer2.5 on an RTX 5090" \
  --client claude-code
rm -rf "$TMP"
