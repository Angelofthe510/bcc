#!/bin/bash
# ─────────────────────────────────────────────
#  Coqui TTS Studio — Backend Launcher
#  Double-click this file in Finder to start.
# ─────────────────────────────────────────────

# Move to the folder this script lives in
cd "$(dirname "$0")"

echo "============================================"
echo "  🐸 Coqui TTS Studio — Starting backend"
echo "============================================"
echo ""
echo "  Once running, open index.html in your browser."
echo "  Then click ⚡ GO ONLINE to connect."
echo ""
echo "  Press Ctrl+C here to stop the server."
echo "--------------------------------------------"
echo ""

# Accept license silently
export COQUI_TOS_AGREED=1

python3 backend.py
