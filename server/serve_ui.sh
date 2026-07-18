#!/usr/bin/env bash
# サヨ子対話UIサーバを起動 (g24)。run.bat から ssh 経由で呼ばれる。
# ssh セッションが切れるとこのプロセスも終わる (後片付け不要の設計)。
set -euo pipefail
pkill -f "[m]oshi.server" 2>/dev/null || true
pkill -f "[s]ayoko_ui_server" 2>/dev/null || true
sleep 1
source ~/venvs/moshi-infer-g24/bin/activate
exec python ~/sayoko-fullduplex/ui/server/sayoko_ui_server.py
