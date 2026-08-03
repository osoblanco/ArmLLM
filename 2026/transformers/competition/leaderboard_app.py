"""Streamlit leaderboard + submission portal for the LLaMA-MoE competition.

Students submit their team name, benchmark perplexity, and the code file they
ran. Submissions are stored in SQLite (plus the raw code files on disk for
audit); the leaderboard ranks every team's best perplexity against the
reference baseline. Uploaded code is stored, never executed.

Run (reachable from outside on port 8501):
  streamlit run leaderboard_app.py --server.address 0.0.0.0 \
      --server.port 8501 --server.headless true

Data lives outside the repo in ~/moe_competition_data/ .
Organizers: edit ~/moe_competition_data/benchmarks.json to set baselines.
"""

import json
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path.home() / "moe_competition_data"
CODE_DIR = DATA_DIR / "code"
DB_PATH = DATA_DIR / "submissions.db"
BENCHMARKS_PATH = DATA_DIR / "benchmarks.json"

MAX_CODE_BYTES = 1_000_000  # 1 MB is plenty for a single .py file
MIN_PPL = 1.0  # perplexity is mathematically >= 1
MAX_PPL = 60_000.0  # ~ vocab size: an untrained model scores ~50k


def init_storage():
    DATA_DIR.mkdir(exist_ok=True)
    CODE_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submitted_at TEXT NOT NULL,
                team TEXT NOT NULL,
                members TEXT,
                perplexity REAL NOT NULL,
                code_file TEXT NOT NULL,
                notes TEXT
            )"""
        )
    if not BENCHMARKS_PATH.exists():
        BENCHMARKS_PATH.write_text(json.dumps([], indent=2))


def load_benchmarks():
    try:
        return json.loads(BENCHMARKS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def load_submissions() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT team, members, perplexity, submitted_at FROM submissions", conn
        )


def save_submission(team, members, perplexity, code_bytes, code_name, notes):
    safe_team = re.sub(r"[^A-Za-z0-9_-]+", "_", team)[:40]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stored_name = f"{safe_team}_{stamp}_{re.sub(r'[^A-Za-z0-9._-]+', '_', code_name)}"
    (CODE_DIR / stored_name).write_bytes(code_bytes)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO submissions (submitted_at, team, members, perplexity, code_file, notes)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                team.strip(),
                members.strip(),
                float(perplexity),
                stored_name,
                notes.strip(),
            ),
        )


def build_leaderboard() -> pd.DataFrame:
    """Best (lowest) perplexity per team, merged with the benchmark rows."""
    rows = []
    df = load_submissions()
    if not df.empty:
        best = df.loc[df.groupby("team")["perplexity"].idxmin()]
        for _, r in best.iterrows():
            n_subs = int((df["team"] == r["team"]).sum())
            rows.append(
                {
                    "Team": r["team"],
                    "Perplexity": r["perplexity"],
                    "Submissions": n_subs,
                    "Last submitted": r["submitted_at"],
                    "_benchmark": False,
                }
            )
    for b in load_benchmarks():
        rows.append(
            {
                "Team": f"⭐ {b['name']}",
                "Perplexity": b["perplexity"],
                "Submissions": "—",
                "Last submitted": b.get("date", "—"),
                "_benchmark": True,
            }
        )
    if not rows:
        return pd.DataFrame()
    board = pd.DataFrame(rows).sort_values("Perplexity").reset_index(drop=True)
    board.insert(0, "Rank", board.index + 1)
    return board


st.set_page_config(page_title="LLaMA-MoE Competition", page_icon="🏆", layout="wide")
init_storage()

st.title("🏆 LLaMA + Mixture-of-Experts Competition")
st.caption(
    "Benchmark: perplexity on the WikiText-2 (raw) validation split, GPT-2 "
    "tokenizer, 128-token chunks — the number printed by `python <file>.py eval`. "
    "Lower is better."
)

board_tab, submit_tab, rules_tab = st.tabs(["Leaderboard", "Submit", "Rules"])

with board_tab:
    board = build_leaderboard()
    if board.empty:
        st.info("No submissions yet — be the first!")
    else:
        benchmarks = board[board["_benchmark"]]
        if not benchmarks.empty:
            baseline = benchmarks["Perplexity"].min()
            board["vs baseline"] = board["Perplexity"].apply(
                lambda p: f"{(p - baseline) / baseline * 100:+.1f}%"
            )
        st.dataframe(
            board.drop(columns=["_benchmark"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("⭐ rows are organizer baselines. Teams are ranked by their best submission.")

with submit_tab:
    st.subheader("Submit your result")
    with st.form("submission", clear_on_submit=True):
        team = st.text_input("Team name *", max_chars=60)
        members = st.text_input("Team members (names / emails)", max_chars=200)
        perplexity = st.number_input(
            "Validation perplexity (from `eval` mode) *",
            min_value=MIN_PPL,
            max_value=MAX_PPL,
            value=100.0,
            step=0.01,
            format="%.2f",
        )
        code_file = st.file_uploader(
            "Your code file (.py) — the exact file you ran *", type=["py"]
        )
        notes = st.text_area(
            "Notes (what you changed vs the handout)", max_chars=1000, height=80
        )
        submitted = st.form_submit_button("Submit")

    if submitted:
        errors = []
        if not team.strip():
            errors.append("Team name is required.")
        if code_file is None:
            errors.append("Code file is required.")
        elif code_file.size > MAX_CODE_BYTES:
            errors.append("Code file too large (max 1 MB).")
        if errors:
            for e in errors:
                st.error(e)
        else:
            save_submission(
                team, members, perplexity, code_file.getvalue(), code_file.name, notes
            )
            st.success(
                f"Submission recorded for **{team.strip()}**: perplexity "
                f"{perplexity:.2f}. Check the leaderboard tab!"
            )
            st.balloons()

with rules_tab:
    st.markdown(
        """
### Rules

1. **Fixed tokenizer & vocab** — GPT-2, `vocab_size = 50257`.
2. **Fixed data** — train ONLY on the WikiText-2 raw `train` split, at most
   **3 epochs**. No external data, no pretrained weights, no distillation.
   The validation split must never be used for training.
3. **Parameter budgets** — active params per token ≤ **35M**, total params
   ≤ **120M** (`check_constraints()` in the handout enforces this).
4. **Reproducibility** — fixed seed 42; your number must reproduce by
   running `train` then `eval` on the file you upload here.
5. **Frozen eval pipeline** — competition constants, `evaluate()`,
   `load_and_preprocess_data()`, the `eval` mode block, and
   `count_params`/`check_constraints` must remain unmodified.
6. **Single GPU.**

Uploaded code is stored for **audit** — organizers re-run submissions to
verify reported numbers. A submission that does not reproduce or violates
the rules is disqualified. You may resubmit as often as you like; your best
score counts.
"""
    )
