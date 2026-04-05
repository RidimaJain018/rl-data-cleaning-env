"""
gradio_ui.py — Gradio demo UI for DataCleaningEnv v1.2
=======================================================
Tab 1: Built-in Dataset  — easy / medium / hard (10 columns, up to 50 rows)
Tab 2: Upload Your CSV   — any CSV, any columns, baseline auto-cleans it

Deployment
----------
HF Spaces Docker runtime:
  - FastAPI runs on port 8000  (OpenEnv API, handles /reset /step /state etc.)
  - Gradio  runs on port 7860  (demo UI, routed by HF Spaces automatically)

Run locally:
  python gradio_ui.py        # Gradio only, http://localhost:7860
  python app.py              # FastAPI only, http://localhost:8000
  ./start.sh                 # Both together (production / Spaces mode)
"""

import io
import os

import gradio as gr
import pandas as pd
from env import DataCleaningEnv
from agent import baseline_agent, upload_agent

# ── Palette ──────────────────────────────────────────────────────────────────
BG        = "#EAE3D5"
SURFACE   = "#2E2B38"
SURFACE2  = "#3C3948"
ROW_ODD   = "#34313F"
ROW_EVEN  = "#2C2A35"
BEIGE     = "#C1B098"
TEAL      = "#6AACB8"
DIRTY_RED = "#C47A72"
TEXT      = "#F0EBE4"
SUBTEXT   = "#A89DB8"
BG_LABEL  = "#7A6E8A"
BORDER    = "#48455A"
BG_BORDER = "#C8C0B0"
PURPLE    = "#534AB7"
LAVENDER  = "#CECBF6"

WAIT_MSG = (
    "<div style='background:" + SURFACE + ";border:1px solid " + BORDER + ";"
    "border-radius:12px;padding:18px 22px;color:" + SUBTEXT + ";"
    "font-family:DM Mono,monospace;font-size:0.82rem;'>"
    "Select options and click Run.</div>"
)
UPLOAD_WAIT = (
    "<div style='background:" + SURFACE + ";border:1px solid " + BORDER + ";"
    "border-radius:12px;padding:18px 22px;color:" + SUBTEXT + ";"
    "font-family:DM Mono,monospace;font-size:0.82rem;'>"
    "Upload a CSV or paste data below, then click Clean My Data.</div>"
)
PLACEHOLDER = (
    "<div style='color:" + SUBTEXT + ";font-family:DM Mono,monospace;"
    "font-size:0.82rem;padding:16px;'>No data yet.</div>"
)


# ── LLM agent factory ─────────────────────────────────────────────────────────
def _llm_factory(api_key: str, base_url: str, model: str):
    import json
    import re
    from openai import OpenAI
    client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip() or None)

    def _agent(obs):
        if obs is None:
            return 0
        row = obs["row_data"]
        # Show raw values only — no pre-computed hints so the LLM must reason genuinely
        lines = []
        for col, val in row.items():
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                lines.append(f"  {col}: NULL")
            elif isinstance(val, str):
                lines.append(f'  {col}: "{val}"')
            else:
                lines.append(f"  {col}: {val}")
        try:
            resp = client.chat.completions.create(
                model=model.strip() or "meta-llama/Llama-3.2-3B-Instruct",
                messages=[
                    {"role": "system", "content":
                     "You are an expert data quality analyst reviewing rows from an employee dataset.\n\n"
                     "DATASET SCHEMA AND VALID RANGES:\n"
                     "  age              : integer, valid range 22-62\n"
                     "  salary           : integer USD, valid range 35000-180000\n"
                     "  city             : string, one of [NY, LA, SF, Chicago, Austin, Seattle, Boston, Denver]\n"
                     "  experience       : integer years, valid range 0-30\n"
                     "  rating           : float, valid range 0.0-5.0\n"
                     "  department       : string, one of [Engineering, Sales, HR, Marketing, Finance, Operations]\n"
                     "  bonus            : integer USD, valid range 0-25000\n"
                     "  years_at_company : integer years, valid range 0-20\n"
                     "  performance_score: float, valid range 0.0-5.0\n"
                     "  overtime_hours   : integer, valid range 0-80\n\n"
                     "ACTIONS:\n"
                     "  0 = skip           — row has no data quality issue\n"
                     "  1 = impute_missing — row has a NULL / missing cell\n"
                     "  3 = fix_outlier    — row has any other issue (outlier, invalid value,\n"
                     "                       type mismatch, whitespace padding, negative)\n\n"
                     "Carefully examine each field against the valid ranges. "
                     'Reply ONLY with JSON: {"action": <0|1|3>}'},
                    {"role": "user", "content":
                     "Examine this employee record and identify any data quality issue:\n\n"
                     + "\n".join(lines)
                     + "\n\nCheck each field against the valid ranges. "
                     'Reply ONLY with JSON: {"action": <0|1|3>}'},
                ],
                temperature=0, max_tokens=32,
            )
            raw = resp.choices[0].message.content.strip()
            try:
                a = int(json.loads(raw)["action"])
            except Exception:
                m = re.search(r'"action"\s*:\s*(\d)', raw)
                a = int(m.group(1)) if m else 0
            return a if a in {0, 1, 3} else 0
        except Exception as e:
            print(f"[llm] {e} — baseline fallback")
            return baseline_agent(obs)

    return _agent


# ── HTML table builder ────────────────────────────────────────────────────────
def df_to_html(df, max_cols: int = 12) -> str:
    if df is None or (hasattr(df, "empty") and df.empty):
        return PLACEHOLDER
    cols      = list(df.columns)
    truncated = len(cols) > max_cols
    show_cols = cols[:max_cols]

    th = (
        "background:" + SURFACE2 + ";color:" + SUBTEXT + ";"
        "font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.12em;"
        "text-transform:uppercase;padding:11px 13px;text-align:left;"
        "border-bottom:2px solid " + BG + ";white-space:nowrap;"
    )
    heads = "".join(f"<th style='{th}'>{c}</th>" for c in show_cols)
    if truncated:
        heads += f"<th style='{th}'>+{len(cols) - max_cols} more</th>"

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg    = ROW_ODD if i % 2 == 0 else ROW_EVEN
        cells = ""
        for col in show_cols:
            val = row[col]
            s   = str(val) if val is not None else "null"
            if s in ("null", "nan", "None"):                              color = DIRTY_RED
            elif s == "OK":                                               color = TEAL
            elif s == "X":                                                color = DIRTY_RED
            elif s.startswith("+") and s[1:].replace(".", "").isdigit(): color = TEAL
            elif s.startswith("-") and s[1:].replace(".", "").isdigit(): color = DIRTY_RED
            else:                                                         color = TEXT
            cells += (
                f"<td style='background:{bg};color:{color};"
                "font-family:DM Mono,monospace;font-size:0.8rem;"
                f"padding:9px 13px;border-bottom:1px solid {BORDER}44;'>{s}</td>"
            )
        if truncated:
            cells += (
                f"<td style='background:{bg};color:{SUBTEXT};"
                "font-family:DM Mono,monospace;font-size:0.7rem;"
                f"padding:9px 13px;'>…</td>"
            )
        rows_html += f"<tr>{cells}</tr>"

    return (
        f"<div style='border-radius:12px;overflow:auto;border:1px solid {BORDER};"
        f"background:{ROW_ODD};'>"
        "<table style='width:100%;border-collapse:collapse;min-width:500px;'>"
        f"<thead><tr>{heads}</tr></thead><tbody>{rows_html}</tbody></table></div>"
    )


# ── Metric cards ──────────────────────────────────────────────────────────────
def _card(top, big, bot, color=None):
    c = color or TEXT
    return (
        f"<div style='background:{SURFACE};border:1px solid {BORDER};"
        "border-radius:12px;padding:18px 20px;'>"
        f"<div style='font-size:0.55rem;font-family:DM Mono,monospace;"
        f"letter-spacing:0.16em;text-transform:uppercase;color:{TEAL};"
        f"margin-bottom:6px;'>{top}</div>"
        f"<div style='font-size:2.1rem;font-weight:700;color:{c};"
        f"letter-spacing:-0.04em;line-height:1;'>{big}</div>"
        f"<div style='font-size:0.62rem;color:{SUBTEXT};"
        f"margin-top:7px;font-family:DM Mono,monospace;'>{bot}</div></div>"
    )


def _summary(label, score, steps, reward, fixed, total, agent_name="baseline"):
    pct   = str(round(score * 100)) + "%"
    rew_s = ("+" if reward > 0 else "") + str(reward)
    badge = (
        f"<span style='background:{PURPLE};color:{LAVENDER};"
        "font-size:0.58rem;font-family:DM Mono,monospace;"
        "padding:3px 10px;border-radius:20px;text-transform:uppercase;"
        f"letter-spacing:0.1em;'>{agent_name.upper()} AGENT</span>"
    )
    fixed_str = (
        str(fixed)
        + f"<span style='font-size:0.95rem;color:{SUBTEXT};font-weight:400;'> / {total}</span>"
    )
    cards = (
        _card(label.upper() + " TASK", pct, "Score", BEIGE)
        + _card("Steps", str(steps), "Taken this episode")
        + _card("Reward", rew_s, "Total reward", TEAL)
        + _card("Fixed", fixed_str, "Issues resolved")
    )
    return (
        f"<div style='margin-bottom:10px;'>{badge}</div>"
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'>"
        + cards + "</div>"
    )


def _summary_all(rows, agent_name="baseline"):
    badge = (
        f"<span style='background:{PURPLE};color:{LAVENDER};"
        "font-size:0.58rem;font-family:DM Mono,monospace;"
        "padding:3px 10px;border-radius:20px;text-transform:uppercase;"
        f"letter-spacing:0.1em;'>{agent_name.upper()} AGENT</span>"
    )
    cards = "".join(
        _card(t.upper(), str(round(s * 100)) + "%",
              f"{f}/{n} fixed · {st} steps · {('+' if r > 0 else '')}{r}", BEIGE)
        for t, s, st, r, f, n in rows
    )
    return (
        f"<div style='margin-bottom:10px;'>{badge}</div>"
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;'>"
        + cards + "</div>"
    )


# ── Shared helpers ────────────────────────────────────────────────────────────
def _trace_df(env) -> pd.DataFrame:
    if not env.episode_log:
        return pd.DataFrame()
    t = pd.DataFrame(env.episode_log)[
        ["step", "row", "col", "issue", "action", "correct", "old_value", "new_value", "reward"]
    ]
    t["correct"] = t["correct"].map({True: "OK", False: "X"})
    t["reward"]  = t["reward"].apply(lambda r: ("+" if r > 0 else "") + str(int(r)))
    return t


def _resolve_agent(agent_type, api_key, base_url, model):
    if agent_type == "LLM Agent" and api_key.strip():
        try:
            return _llm_factory(api_key, base_url, model), "llm"
        except Exception as e:
            print(f"[UI] LLM init failed: {e}")
    return baseline_agent, "baseline"


# ── Runner: single built-in task ─────────────────────────────────────────────
def run_task(task_level, agent_type, api_key, base_url, model):
    task_level = (task_level or "medium").lower()
    agent_fn, agent_name = _resolve_agent(agent_type, api_key, base_url, model)
    env  = DataCleaningEnv()
    obs  = env.reset(task_level=task_level)
    rew  = 0
    done = False
    while not done:
        obs, r, done, _ = env.step(agent_fn(obs))
        rew += r
    fixed = sum(1 for e in env.episode_log if e["correct"])
    return (
        _summary(task_level, env.grade(), env.steps, rew, fixed, env.total_issues_at_start, agent_name),
        df_to_html(env.original_df),
        df_to_html(env.df),
        df_to_html(_trace_df(env)),
    )


# ── Runner: all three built-in levels ────────────────────────────────────────
def run_all(agent_type, api_key, base_url, model):
    agent_fn, agent_name = _resolve_agent(agent_type, api_key, base_url, model)
    rows, traces, last_env = [], [], None
    for task in ["easy", "medium", "hard"]:
        env  = DataCleaningEnv()
        obs  = env.reset(task_level=task)
        rew  = 0
        done = False
        while not done:
            obs, r, done, _ = env.step(agent_fn(obs))
            rew += r
        fixed = sum(1 for e in env.episode_log if e["correct"])
        rows.append((task, env.grade(), env.steps, rew, fixed, env.total_issues_at_start))
        t = _trace_df(env)
        if not t.empty:
            t.insert(0, "task", task)
            traces.append(t)
        last_env = env
    combined = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    return (
        _summary_all(rows, agent_name),
        df_to_html(last_env.original_df),
        df_to_html(last_env.df),
        df_to_html(combined),
    )


# ── Runner: user CSV upload ───────────────────────────────────────────────────
def run_upload(file_obj, csv_text):
    """
    Clean a user-provided CSV file or pasted CSV text using the upload agent.
    Works with any column names and any number of rows.
    """
    text = None
    if file_obj is not None:
        try:
            raw = open(file_obj.name, "rb").read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
        except Exception as e:
            return (
                f"<div style='color:{DIRTY_RED};font-family:DM Mono,monospace;"
                f"padding:12px;'>File read error: {e}</div>",
                PLACEHOLDER, PLACEHOLDER, PLACEHOLDER,
            )
    elif csv_text and csv_text.strip():
        text = csv_text.strip()

    if not text:
        return UPLOAD_WAIT, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        return (
            f"<div style='color:{DIRTY_RED};font-family:DM Mono,monospace;"
            f"padding:12px;'>CSV parse error: {e}</div>",
            PLACEHOLDER, PLACEHOLDER, PLACEHOLDER,
        )
    if df.empty:
        return (
            f"<div style='color:{DIRTY_RED};font-family:DM Mono,monospace;"
            f"padding:12px;'>CSV is empty.</div>",
            PLACEHOLDER, PLACEHOLDER, PLACEHOLDER,
        )

    env = DataCleaningEnv()
    try:
        obs = env.reset_from_dataframe(df)
    except Exception as e:
        return (
            f"<div style='color:{DIRTY_RED};font-family:DM Mono,monospace;"
            f"padding:12px;'>Error: {e}</div>",
            PLACEHOLDER, PLACEHOLDER, PLACEHOLDER,
        )

    rew  = 0
    done = False
    while not done and obs is not None:
        obs, r, done, _ = env.step(upload_agent(obs))
        rew += r

    fixed = sum(1 for e in env.episode_log if e["correct"])
    note = (
        f"<div style='font-size:0.68rem;font-family:DM Mono,monospace;"
        f"color:{SUBTEXT};margin-bottom:10px;letter-spacing:0.06em;'>"
        f"Detected <b style='color:{TEXT}'>{env.total_issues_at_start} issues</b> "
        f"across <b style='color:{TEXT}'>{len(df)} rows</b> and "
        f"<b style='color:{TEXT}'>{len(df.columns)} columns</b>. "
        f"IQR-based outlier detection — works with any column names."
        "</div>"
    )
    return (
        note + _summary("custom", env.grade(), env.steps, rew,
                        fixed, env.total_issues_at_start, "baseline"),
        df_to_html(env.original_df),
        df_to_html(env.df),
        df_to_html(_trace_df(env)),
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
css = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');
*,*::before,*::after{{box-sizing:border-box;}}
body,.gradio-container,.gradio-container>.main,.gradio-container>.main>.wrap,.gradio-container>div{{
    background:{BG}!important;font-family:'DM Sans',sans-serif!important;color:#2A2535!important;}}
.gd-header{{background:{SURFACE};border-bottom:1px solid {BORDER};
    padding:22px 34px 18px;border-radius:0 0 16px 16px;margin-bottom:18px;}}
.gd-title{{font-size:1.6rem;font-weight:700;color:{BEIGE};letter-spacing:-0.025em;margin:0 0 4px;}}
.gd-sub{{font-size:0.65rem;color:{SUBTEXT};letter-spacing:0.1em;text-transform:uppercase;
    font-family:'DM Mono',monospace;margin:0;}}
.sec{{font-size:0.58rem;font-weight:500;letter-spacing:0.18em;text-transform:uppercase;
    font-family:'DM Mono',monospace;color:{BG_LABEL};border-bottom:1px solid {BG_BORDER};
    padding-bottom:8px;margin:16px 0 12px;}}
.pdirty{{font-size:0.6rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
    font-family:'DM Mono',monospace;color:{DIRTY_RED};margin-bottom:7px;}}
.pclean{{font-size:0.6rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
    font-family:'DM Mono',monospace;color:{TEAL};margin-bottom:7px;}}
.upload-hint{{font-size:0.7rem;font-family:'DM Mono',monospace;color:{SUBTEXT};
    line-height:1.65;margin:10px 0 14px;}}
#btn-task,#btn-all{{display:none!important;}}
.gradio-html{{background:transparent!important;border:none!important;padding:0!important;}}
"""

# ── JS constants ──────────────────────────────────────────────────────────────
_SEL = (
    f"background:{SURFACE};color:{TEXT};border:1px solid {BORDER};border-radius:10px;"
    "padding:11px 14px;font-family:DM Mono,monospace;font-size:0.84rem;"
    "cursor:pointer;outline:none;width:100%;-webkit-appearance:none;appearance:none;"
    "background-image:url('data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 "
    "width=%2212%22 height=%228%22><path d=%22M2 2l4 4 4-4%22 stroke=%22%23A89DB8%22 "
    "stroke-width=%221.5%22 fill=%22none%22 stroke-linecap=%22round%22/></svg>');"
    "background-repeat:no-repeat;background-position:right 14px center;"
)
_OPT = f"background:{SURFACE};color:{TEXT};"
_INP = (
    f"background:{SURFACE};color:{TEXT};border:1px solid {BORDER};border-radius:10px;"
    "padding:10px 14px;font-family:DM Mono,monospace;font-size:0.8rem;outline:none;width:100%;"
)
_LBL = (
    f"font-size:0.58rem;font-family:DM Mono,monospace;letter-spacing:0.13em;"
    f"text-transform:uppercase;color:{BG_LABEL};margin-bottom:4px;display:block;"
)
_BTN_P = (
    f"width:100%;background:{BEIGE};color:#1A1720;border:none;border-radius:10px;"
    "padding:12px 14px;font-family:DM Mono,monospace;font-weight:700;font-size:0.76rem;"
    "letter-spacing:0.08em;text-transform:uppercase;cursor:pointer;"
)
_BTN_S = (
    f"width:100%;background:{SURFACE};color:{BEIGE};border:1px solid {BORDER};"
    "border-radius:10px;padding:12px 14px;font-family:DM Mono,monospace;"
    "font-weight:600;font-size:0.76rem;letter-spacing:0.08em;text-transform:uppercase;cursor:pointer;"
)
_JS_TASK  = "var t=document.querySelector('#ht textarea')||document.querySelector('#ht input');if(t){t.value=this.value;t.dispatchEvent(new Event('input',{bubbles:true}));}"
_JS_AGENT = f"var t=document.querySelector('#ha textarea')||document.querySelector('#ha input');if(t){{t.value=this.value;t.dispatchEvent(new Event('input',{{bubbles:true}}));}}var p=document.getElementById('llmp');if(p){{p.style.display=this.value==='LLM Agent'?'block':'none';}}"
_JS_KEY   = "var t=document.querySelector('#hk textarea')||document.querySelector('#hk input');if(t){t.value=this.value;t.dispatchEvent(new Event('input',{bubbles:true}));}"
_JS_URL   = "var t=document.querySelector('#hu textarea')||document.querySelector('#hu input');if(t){t.value=this.value;t.dispatchEvent(new Event('input',{bubbles:true}));}"
_JS_MDL   = "var t=document.querySelector('#hm textarea')||document.querySelector('#hm input');if(t){t.value=this.value;t.dispatchEvent(new Event('input',{bubbles:true}));}"

_DEFAULT_URL   = "https://api-inference.huggingface.co/v1"
_DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

CTRL_HTML = (
    f"<div style='display:flex;align-items:flex-end;gap:10px;margin-bottom:8px;'>"
    f"<div style='flex:2;display:flex;flex-direction:column;gap:4px;'>"
    f"<span style='{_LBL}'>Task Difficulty</span>"
    f"<select onchange=\"{_JS_TASK}\" style='{_SEL}'>"
    f"<option value='easy' style='{_OPT}'>Easy — 20 rows, 8 issues</option>"
    f"<option value='medium' selected style='{_OPT}'>Medium — 30 rows, 12 issues</option>"
    f"<option value='hard' style='{_OPT}'>Hard — 50 rows, 20 issues</option>"
    f"</select></div>"
    f"<div style='flex:2;display:flex;flex-direction:column;gap:4px;'>"
    f"<span style='{_LBL}'>Agent</span>"
    f"<select onchange=\"{_JS_AGENT}\" style='{_SEL}'>"
    f"<option value='Baseline Agent' selected style='{_OPT}'>Baseline Agent</option>"
    f"<option value='LLM Agent' style='{_OPT}'>LLM Agent</option>"
    f"</select></div>"
    f"<div style='flex:1;'><button onclick=\"setTimeout(()=>document.querySelector('#btn-task button').click(),50)\" style='{_BTN_P}'>&#9654; Run Task</button></div>"
    f"<div style='flex:1;'><button onclick=\"setTimeout(()=>document.querySelector('#btn-all button').click(),50)\" style='{_BTN_S}'>&#9889; Run All</button></div>"
    f"</div>"
    f"<div id='llmp' style='display:none;background:{SURFACE};border:1px solid {BORDER};"
    "border-radius:12px;padding:14px 18px;margin-bottom:8px;'>"
    f"<div style='font-size:0.56rem;font-weight:500;letter-spacing:0.14em;text-transform:uppercase;"
    f"font-family:DM Mono,monospace;color:{LAVENDER};margin-bottom:10px;'>&#9670; LLM Configuration</div>"
    f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;'>"
    f"<div><label style='{_LBL}'>API Key (HF_TOKEN)</label>"
    f"<input type='password' placeholder='hf_...' oninput=\"{_JS_KEY}\" style='{_INP}'/></div>"
    f"<div><label style='{_LBL}'>API Base URL</label>"
    f"<input type='text' value='{_DEFAULT_URL}' oninput=\"{_JS_URL}\" style='{_INP}'/></div>"
    f"<div><label style='{_LBL}'>Model Name</label>"
    f"<input type='text' value='{_DEFAULT_MODEL}' oninput=\"{_JS_MDL}\" style='{_INP}'/></div>"
    f"</div></div>"
)


# ── Layout ────────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="RL Data Cleaning Agent") as demo:

    gr.HTML(
        f"<div class='gd-header'><div style='display:flex;justify-content:space-between;"
        "align-items:flex-start;flex-wrap:wrap;gap:10px;'>"
        "<div><h1 class='gd-title'>RL Data Cleaning Agent</h1>"
        f"<p class='gd-sub'>Meta × Scaler OpenEnv Hackathon &nbsp;·&nbsp; "
        "Reinforcement Learning Environment</p></div>"
        f"<span style='background:{SURFACE2};border:1px solid {BEIGE}33;color:{BEIGE};"
        "font-size:0.6rem;font-family:DM Mono,monospace;letter-spacing:0.1em;"
        "padding:5px 12px;border-radius:20px;text-transform:uppercase;'>v1.2 · 10 columns</span>"
        "</div></div>"
    )

    # Hidden textboxes — bridge JS ↔ Gradio Python
    ht = gr.Textbox(value="medium",         elem_id="ht", visible=False, interactive=True)
    ha = gr.Textbox(value="Baseline Agent", elem_id="ha", visible=False, interactive=True)
    hk = gr.Textbox(value="",              elem_id="hk", visible=False, interactive=True)
    hu = gr.Textbox(value=_DEFAULT_URL,    elem_id="hu", visible=False, interactive=True)
    hm = gr.Textbox(value=_DEFAULT_MODEL,  elem_id="hm", visible=False, interactive=True)

    with gr.Tabs():

        # ── Tab 1: Built-in dataset ───────────────────────────────────────
        with gr.Tab("📊 Built-in Dataset"):
            gr.HTML(CTRL_HTML)
            with gr.Row(visible=False):
                btn_task = gr.Button("t", elem_id="btn-task")
                btn_all  = gr.Button("a", elem_id="btn-all")

            gr.HTML('<div class="sec">Results</div>')
            out_sum = gr.HTML(WAIT_MSG)

            gr.HTML('<div class="sec">Dataset — Before vs After Cleaning</div>')
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML('<div class="pdirty">● Before — Dirty</div>')
                    out_before = gr.HTML(PLACEHOLDER)
                with gr.Column():
                    gr.HTML('<div class="pclean">● After — Cleaned</div>')
                    out_after = gr.HTML(PLACEHOLDER)

            gr.HTML('<div class="sec">Episode Trace — Step-by-Step Decisions</div>')
            out_trace = gr.HTML(PLACEHOLDER)

            btn_task.click(fn=run_task,
                           inputs=[ht, ha, hk, hu, hm],
                           outputs=[out_sum, out_before, out_after, out_trace])
            btn_all.click(fn=run_all,
                          inputs=[ha, hk, hu, hm],
                          outputs=[out_sum, out_before, out_after, out_trace])

        # ── Tab 2: Upload CSV ─────────────────────────────────────────────
        with gr.Tab("📂 Upload Your CSV"):
            gr.HTML(
                "<div class='upload-hint'>"
                "Upload <b>any CSV file</b> — your own data, any columns, any number of rows.<br/>"
                "The baseline agent will automatically detect and fix data quality issues using "
                "column-agnostic rules:<br/>"
                f"<span style='color:{TEAL};'>missing values &nbsp;·&nbsp; "
                "IQR outliers &nbsp;·&nbsp; duplicate rows &nbsp;·&nbsp; "
                "type mismatches &nbsp;·&nbsp; whitespace padding</span>"
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="Upload CSV file", file_types=[".csv"])
                    gr.HTML(
                        f"<div style='text-align:center;color:{SUBTEXT};"
                        "font-size:0.7rem;font-family:DM Mono,monospace;"
                        "margin:6px 0;'>— or paste CSV text below —</div>"
                    )
                    csv_in = gr.Textbox(
                        label="Paste CSV text",
                        placeholder="col1,col2,col3\nval1,,val3\nval4,999999,val6\n...",
                        lines=7, max_lines=14,
                    )
                    clean_btn = gr.Button("🧹 Clean My Data", variant="primary")

            gr.HTML('<div class="sec">Results</div>')
            up_sum = gr.HTML(UPLOAD_WAIT)

            gr.HTML('<div class="sec">Your Data — Before vs After</div>')
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML('<div class="pdirty">● Before — Dirty</div>')
                    up_before = gr.HTML(PLACEHOLDER)
                with gr.Column():
                    gr.HTML('<div class="pclean">● After — Cleaned</div>')
                    up_after = gr.HTML(PLACEHOLDER)

            gr.HTML('<div class="sec">Cleaning Trace</div>')
            up_trace = gr.HTML(PLACEHOLDER)

            clean_btn.click(
                fn=run_upload,
                inputs=[file_in, csv_in],
                outputs=[up_sum, up_before, up_after, up_trace],
            )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Respect GRADIO_SERVER_PORT env var so start.sh can override it.
    # Default: 7860 (Gradio standard, routed automatically by HF Spaces).
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",   # bind to all interfaces — required inside Docker
        server_port=port,
        share=False,             # never create a public tunnel; HF Spaces handles routing
    )
