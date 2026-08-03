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

REPO="$HOME/do_not_touch"
README_REL="2026/transformers/competition/README.md"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# Rewrite the "## Live leaderboard: ..." header in the competition README
# and push it, so students always find the current URL on GitHub.
update_readme_url() {
    local url="$1"
    git -C "$REPO" pull --ff-only origin main >> "$LOG" 2>&1 \
        || log "WARN: git pull failed; updating README from local state"
    sed -i "s|^## Live leaderboard: .*|## Live leaderboard: $url|" "$REPO/$README_REL"
    if ! git -C "$REPO" diff --quiet -- "$README_REL"; then
        git -C "$REPO" add "$README_REL" >> "$LOG" 2>&1
        git -C "$REPO" commit -m "Update live leaderboard URL [watchdog auto-commit]" >> "$LOG" 2>&1
        if git -C "$REPO" push origin main >> "$LOG" 2>&1; then
            log "README URL updated and pushed: $url"
        else
            log "WARN: git push failed — README updated locally only"
        fi
    fi
}

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
            local old_url
            old_url=$(cat "$URL_FILE" 2>/dev/null)
            echo "$url" > "$URL_FILE"
            log "new public URL: $url"
            [ "$url" != "$old_url" ] && update_readme_url "$url"
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
