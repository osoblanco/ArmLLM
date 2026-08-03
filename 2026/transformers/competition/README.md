# Competition Leaderboard

## Live leaderboard: https://diabetes-way-letter-controlled.trycloudflare.com

A Streamlit portal where students submit their team name, benchmark
perplexity, and the code file they ran. Uploaded code is stored (never
executed) so organizers can audit and re-run submissions.

## Production hosting (current setup)

The leaderboard runs on a dedicated public server (`139.59.213.111`):

- **Streamlit** as the non-root `leaderboard` user, managed by a systemd
  unit (`moe-leaderboard.service`, `Restart=always`, enabled at boot),
  bound to `127.0.0.1:8501`
- **nginx** on port 80 reverse-proxying to it (with websocket upgrade
  headers, which Streamlit requires)
- Data in `/home/leaderboard/moe_competition_data/` (SQLite + uploaded
  code + `benchmarks.json`)

To redeploy after changing `leaderboard_app.py`:

```bash
scp leaderboard_app.py root@<server>:/home/leaderboard/app/
ssh root@<server> systemctl restart moe-leaderboard
```

### Fallback: hosting without a public server

`watchdog.sh` runs the app on any machine (no root or open ports needed)
and publishes it through a Cloudflare quick tunnel, restarting either
whenever they go down (single-instance; safe to invoke repeatedly, e.g.
from `~/.bashrc`). Quick-tunnel URLs change on every cloudflared restart —
the current one is written to `~/moe_competition_data/public_url.txt` and
auto-committed into this README's header.

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
