#!/bin/bash
# Watchdog for the LLaMA-MoE competition leaderboard.
# Ensures the Streamlit app serves on :80 and a cloudflared quick tunnel is
# healthy; restarts either when down. Single-instance via flock. Started
# detached (survives logout); relaunched from ~/.bashrc after a reboot.
#
# Logs:   ~/moe_competition_data/watchdog.log
# URL:    ~/moe_competition_data/public_url.txt (rewritten if tunnel restarts)

DATA="$HOME/moe_competition_data"
APP_DIR="$HOME/do_not_touch/2026/transformers/competition"
STREAMLIT="$HOME/.conda/envs/armllm/bin/streamlit"
CLOUDFLARED="$DATA/bin/cloudflared"
LOG="$DATA/watchdog.log"
URL_FILE="$DATA/public_url.txt"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# Single instance. The lock lives in /tmp (local tmpfs): flock is unreliable
# on the network FS that holds $HOME (instances hang inside the lock call),
# and /tmp clears on reboot, which is exactly when a fresh lock is wanted.
# The PID file adds a liveness check as a second guard.
LOCKFILE="/tmp/moe_leaderboard_watchdog.lock"
PIDFILE="/tmp/moe_leaderboard_watchdog.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    exit 0
fi
exec 9> "$LOCKFILE"
flock -n 9 || exit 0
echo $$ > "$PIDFILE"

ensure_streamlit() {
    local code
    code=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' http://localhost:80 || true)
    if [ "$code" != "200" ]; then
        log "streamlit not serving (code=$code) — (re)starting"
        pkill -f "streamlit run leaderboard_app.py" 2>/dev/null
        sleep 2
        # 9>&- releases the lock fd so children don't hold it after we die
        (cd "$APP_DIR" && setsid "$STREAMLIT" run leaderboard_app.py \
            --server.address 0.0.0.0 --server.port 80 --server.headless true \
            >> "$DATA/streamlit.log" 2>&1 9>&- &)
    fi
}

tunnel_fail_count=0

ensure_tunnel() {
    if ! pgrep -f "cloudflared tunnel" > /dev/null; then
        log "cloudflared not running — starting new quick tunnel"
        : > "$DATA/cloudflared.log"
        (setsid "$CLOUDFLARED" tunnel --url http://localhost:80 --no-autoupdate \
            >> "$DATA/cloudflared.log" 2>&1 9>&- &)
        sleep 20
        local url
        url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$DATA/cloudflared.log" | tail -1)
        if [ -n "$url" ]; then
            echo "$url" > "$URL_FILE"
            log "new public URL: $url"
        else
            log "WARN: tunnel started but URL not found in log yet"
        fi
        tunnel_fail_count=0
        return
    fi
    # Process is alive — verify the URL actually serves; restart after 3 fails
    local url code
    url=$(cat "$URL_FILE" 2>/dev/null)
    [ -z "$url" ] && return
    code=$(curl -s --max-time 15 -o /dev/null -w '%{http_code}' "$url" || true)
    if [ "$code" = "200" ]; then
        tunnel_fail_count=0
    else
        tunnel_fail_count=$((tunnel_fail_count + 1))
        log "tunnel check failed (code=$code, count=$tunnel_fail_count)"
        if [ "$tunnel_fail_count" -ge 3 ]; then
            log "tunnel dead — killing cloudflared for restart"
            pkill -f "cloudflared tunnel" 2>/dev/null
            tunnel_fail_count=0
        fi
    fi
}

log "watchdog started (pid $$)"
while true; do
    ensure_streamlit
    ensure_tunnel
    sleep 30
done
