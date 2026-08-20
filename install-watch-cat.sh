#!/usr/bin/env bash
# Install watch_cat as a systemd *user* service that auto-starts at boot.
# Run on the Pi: bash install-watch-cat.sh

set -euo pipefail

UNIT="watch-cat.service"
SRC="$(dirname "$0")/systemd/$UNIT"
DST="$HOME/.config/systemd/user/$UNIT"

mkdir -p "$HOME/.config/systemd/user"
cp "$SRC" "$DST"

systemctl --user daemon-reload
systemctl --user enable --now "$UNIT"

# Let user services run even when nobody is logged in (needed at boot).
loginctl enable-linger "$USER" 2>/dev/null || true

systemctl --user --no-pager status "$UNIT" || true
echo "Enabled. Control with: systemctl --user [start|stop|restart|status] $UNIT"