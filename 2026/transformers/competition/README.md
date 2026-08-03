# Competition Leaderboard

## Live leaderboard: https://unix-command-ronald-questionnaire.trycloudflare.com

(This header is rewritten automatically by `watchdog.sh` whenever the tunnel
URL changes — always check here for the current address.)

A Streamlit portal where students submit their team name, benchmark
perplexity, and the code file they ran. Uploaded code is stored (never
executed) so organizers can audit and re-run submissions.

## Running the server

From this directory, in the class conda env (see root [SETUP.md](../../../SETUP.md)):

```bash
streamlit run leaderboard_app.py --server.address 0.0.0.0 \
    --server.port 8501 --server.headless true
```

Students then open `http://<machine-public-ip>:8501` in a browser. Make sure
port 8501 is open in the machine's firewall / cloud security group.

### Keeping it up (no open ports / no root)

If the machine has no publicly reachable port, use `watchdog.sh` instead of
running streamlit directly. It keeps the app on `localhost:80` and publishes
it through a Cloudflare quick tunnel, restarting either whenever they go
down (single-instance; safe to invoke repeatedly, e.g. from `~/.bashrc`):

```bash
cp watchdog.sh ~/moe_competition_data/watchdog.sh && chmod +x ~/moe_competition_data/watchdog.sh
# put the cloudflared binary at ~/moe_competition_data/bin/cloudflared, then:
setsid ~/moe_competition_data/watchdog.sh < /dev/null > /dev/null 2>&1 &
```

The current public URL is always in `~/moe_competition_data/public_url.txt`
(quick-tunnel URLs change whenever cloudflared restarts — re-announce it to
students if that happens; `watchdog.log` records every restart).

## Data

Everything is stored outside the repo in `~/moe_competition_data/`:

- `submissions.db` — SQLite table of all submissions
- `code/` — every uploaded code file, timestamped, for audit
- `benchmarks.json` — organizer-set baseline rows shown with a ⭐ on the
  leaderboard. Format:

  ```json
  [
    {"name": "Reference solution (top-2/4 MoE)", "perplexity": 123.45, "date": "2026-08-03"}
  ]
  ```

## Auditing a submission

```bash
python <submitted_file>.py train   # trains with the fixed seed
python <submitted_file>.py eval    # must print the submitted perplexity
```

Also diff the frozen sections (competition constants, `evaluate`,
`load_and_preprocess_data`, `eval` mode, `count_params`/`check_constraints`)
against the handout.
