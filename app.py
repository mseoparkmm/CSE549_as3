# app.py — A3-1 Study (LogiQA format, PID required, Sync AI elimination, Google Sheets save)
# v2.2 — participant-focused
#   - Setup: no cond URL tip; PID field recommends using uniqname
#   - Answer UI: no dummy option; no default selection
#   - Finish: single primary action that writes to provided Google Sheet (no fallback)
#   - Robust secrets handling (AttrDict → dict), Sheets perm hint if denied

import os
import re
import json
import time
import random
import string
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="AI-Assisted Task Study", layout="centered")

DATA_PATH = "logiqa_eval.csv"
FIXED_N = 5
ELIMS_TO_REMOVE = 2
LETTER = ["A", "B", "C", "D"]
DEBUG = False  # 필요시 True

def idx2letter(i: int) -> str: return LETTER[i]
def letter2idx(ch: str) -> int: return LETTER.index(ch)

# ----------------------------
# Secrets / Envs
# ----------------------------
def get_sheet_id() -> str:
    """Return provided SHEETS_DOC_ID from secrets or env (stripped)."""
    try:
        if "sheet" in st.secrets and "SHEETS_DOC_ID" in st.secrets["sheet"]:
            return str(st.secrets["sheet"]["SHEETS_DOC_ID"]).strip()
    except FileNotFoundError:
        pass
    return os.environ.get("SHEETS_DOC_ID", "").strip()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", OPENAI_API_KEY)
except FileNotFoundError:
    pass
USE_OPENAI = bool(OPENAI_API_KEY)

# ----------------------------
# Query params (new/old Streamlit both)
# ----------------------------
def get_query_params() -> Dict[str, str]:
    if hasattr(st, "query_params"):
        try:
            return dict(st.query_params)
        except Exception:
            pass
    try:
        qp = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) and v else "") for k, v in qp.items()}
    except Exception:
        return {}

# ----------------------------
# Data loading
# ----------------------------
def safe_json_load(s):
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(str(s).replace("’", "'").replace("“", '"').replace("”", '"'))
        except Exception:
            txt = str(s)
            if "||" in txt:
                return [p.strip() for p in txt.split("||") if p.strip()]
            return [txt]

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"❌ '{path}' not found.")
        st.stop()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")

    cols = {c.lower().strip(): c for c in df.columns}
    for req in ["context", "question"]:
        if req not in cols:
            st.error(f"❌ Required column '{req}' missing. Found: {list(df.columns)}")
            st.stop()

    has_choices = "choices" in cols
    has_abcd = all(k in df.columns for k in ["A", "B", "C", "D"])
    if not has_choices and not has_abcd:
        st.error(f"❌ Need 'choices'(JSON) or A,B,C,D columns. Found: {list(df.columns)}")
        st.stop()
    if "correct_answer" not in cols:
        st.error("❌ 'correct_answer' missing.")
        st.stop()

    df_norm = pd.DataFrame()
    df_norm["context"] = df[cols["context"]].astype(str)
    df_norm["question"] = df[cols["question"]].astype(str)
    if has_choices:
        df_norm["choices"] = df[cols["choices"]].apply(safe_json_load)
    else:
        df_norm["choices"] = df.apply(lambda r: [str(r["A"]), str(r["B"]), str(r["C"]), str(r["D"])], axis=1)
    df_norm["correct_answer"] = df[cols["correct_answer"]]

    fixed_choices = []
    for i, lst in enumerate(df_norm["choices"]):
        if not isinstance(lst, list) or len(lst) < 4:
            st.error(f"❌ Row {i}: need exactly 4 choices. Parsed: {lst!r}")
            st.stop()
        if len(lst) > 4:
            lst = lst[:4]
        fixed_choices.append(lst)
    df_norm["choices"] = fixed_choices

    def to_idx(v):
        try:
            i = int(v); return max(0, min(3, i))
        except Exception:
            L = str(v).strip().upper()
            if L and L[0] in "ABCD": return "ABCD".index(L[0])
            raise ValueError("correct_answer must be 0..3 or A..D")

    df_norm["correct_answer"] = df_norm["correct_answer"].apply(to_idx)
    return df_norm

def pick_sample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if len(df) < k:
        st.error(f"Dataset has only {len[df]} rows; need at least {k}.")
        st.stop()
    return df.sample(k, random_state=None).reset_index(drop=True)

# ----------------------------
# OpenAI elimination (sync)
# ----------------------------
def call_openai_elimination(context: str, question: str, choices: List[str]) -> Tuple[List[int], Dict[str, str]]:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not found.")
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
You assist on a 4-option multiple-choice logical reasoning problem.
Identify EXACTLY {ELIMS_TO_REMOVE} options that are VERY LIKELY INCORRECT and give brief reasons.
Return JSON only:
{{"eliminate":["A","B"],"rationale":{{"A":"reason","B":"reason"}}}}

Passage:
{context}

Question:
{question}

Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
""".strip()

    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=30,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                lines = raw.splitlines()
                if lines and lines[0].lower().startswith("json"):
                    raw = "\n".join(lines[1:])
            data = json.loads(raw)

            elim_letters = data.get("eliminate", [])
            rat = data.get("rationale", {})

            elim_idxs, seen = [], set()
            for ch in elim_letters:
                up = str(ch).strip().upper()
                if up in LETTER:
                    i = letter2idx(up)
                    if i not in seen:
                        elim_idxs.append(i)
                        seen.add(i)
            elim_idxs = elim_idxs[:ELIMS_TO_REMOVE]

            rationales = {}
            for ch in elim_letters:
                up = str(ch).strip().upper()
                if up in LETTER and letter2idx(up) in elim_idxs:
                    rationales[up] = str(rat.get(up, "")).strip() or "Eliminated as implausible."
            return elim_idxs, rationales
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))
    raise last_err

def random_eliminate_two(choices: List[str]) -> Tuple[List[int], Dict[str, str]]:
    rng = random.SystemRandom()
    idxs = list(range(len(choices))); rng.shuffle(idxs)
    elim = idxs[:ELIMS_TO_REMOVE]
    rationales = {idx2letter(i): f"Option {idx2letter(i)} appears inconsistent with key details." for i in elim}
    return elim, rationales

# ----------------------------
# App state
# ----------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "setup"
if "sample" not in st.session_state:
    st.session_state.sample = None
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "logs" not in st.session_state:
    st.session_state.logs = []
if "tic" not in st.session_state:
    st.session_state.tic = None
if "elim_cache" not in st.session_state:
    st.session_state.elim_cache = {}

_qp = get_query_params()
if "participant_id" not in st.session_state:
    st.session_state.participant_id = (_qp.get("pid") or "").strip()
if "condition" not in st.session_state:
    st.session_state.condition = (_qp.get("cond") or "with").lower()

if "ai_helpfulness_overall" not in st.session_state:
    st.session_state.ai_helpfulness_overall = None
if "completion_code" not in st.session_state:
    st.session_state.completion_code = None
if "finished" not in st.session_state:
    st.session_state.finished = False  # after pressing Finish & success

if st.session_state.participant_id == "" and st.session_state.stage != "setup":
    st.session_state.stage = "setup"

# ----------------------------
# Header
# ----------------------------
st.title("Human-Subjects Study: AI-Assisted Task (LogiQA)")
cond_badge = "WITH AI" if st.session_state.condition == "with" else "WITHOUT AI"
st.caption(f"Condition: **{cond_badge}** | Participant: **{st.session_state.participant_id or 'N/A'}** | Trials: {FIXED_N}")

if DEBUG:
    def _mask(s, keep=6): s=str(s or ""); return s[:keep]+"…" if len(s)>keep else s
    st.write("— DEBUG — cwd:", os.getcwd())
    p = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
    st.write("— DEBUG — secrets exists?:", os.path.exists(p), p)
    try: st.write("— DEBUG — st.secrets keys:", list(st.secrets.keys()))
    except Exception as e: st.write("— DEBUG — st.secrets error:", repr(e))
    try: sid = get_sheet_id(); st.write("— DEBUG — get_sheet_id():", bool(sid), _mask(sid))
    except Exception as e: st.write("— DEBUG — get_sheet_id error:", repr(e))
    try: _gsa = st.secrets.get("gcp_service_account", None); st.write("— DEBUG — gsa present?:", bool(_gsa))
    except Exception as e: st.write("— DEBUG — gsa error:", repr(e))

# ----------------------------
# Setup
# ----------------------------
if st.session_state.stage == "setup":
    st.subheader("You will solve 5 problems by reading a passage and choosing one answer from four options, with or without AI assistance.")
    st.subheader("For each of the 5 questions, please select your answer and indicate your confidence level (1 to 5).")
    st.subheader("Participant Setup")
    pid = st.text_input(
        "Recommended: use your MTurk worker ID or umich uniqname (e.g., mseopark).",
        value=st.session_state.participant_id,
        placeholder="e.g., uniqname01",
    )
    st.session_state.participant_id = pid.strip()

    if st.button("Start Study", type="primary", disabled=(st.session_state.participant_id == "")):
        df_all = load_dataset(DATA_PATH)
        st.session_state.sample = pick_sample(df_all, FIXED_N)
        st.session_state.idx = 0
        st.session_state.logs = []
        st.session_state.elim_cache = {}
        st.session_state.stage = "running"
        st.session_state.tic = time.time()
        st.rerun()

# ----------------------------
# Running
# ----------------------------
if st.session_state.stage == "running":
    df = st.session_state.sample
    qidx = st.session_state.idx
    row = df.iloc[qidx]

    context = str(row["context"])
    question = str(row["question"])
    choices = list(row["choices"])
    correct_idx = int(row["correct_answer"])

    st.subheader(f"Question {qidx+1} / {FIXED_N}")

    with st.expander("Passage", expanded=True):
        st.write(context)
    st.markdown("**Question**")
    st.write(question)

    if st.session_state.condition == "with":
        if qidx not in st.session_state.elim_cache:
            with st.spinner("🤔 AI is thinking…"):
                try:
                    if USE_OPENAI:
                        elim_idxs, rationales = call_openai_elimination(context, question, choices)
                    else:
                        elim_idxs, rationales = random_eliminate_two(choices)
                except Exception:
                    elim_idxs, rationales = random_eliminate_two(choices)
                st.session_state.elim_cache[qidx] = (elim_idxs, rationales)

        elim_idxs, rationales = st.session_state.elim_cache[qidx]
        st.markdown("**AI eliminations (advice only; all options remain selectable):**")
        for e in elim_idxs:
            L = idx2letter(e)
            st.write(f"- {L}: {rationales.get(L, 'Eliminated as implausible.')}")

    # Answer UI (no dummy option & no default selection)
    col1, col2 = st.columns([3, 2], vertical_alignment="top")
    with col1:
        option_labels = [
            f"A) {choices[0]}",
            f"B) {choices[1]}",
            f"C) {choices[2]}",
            f"D) {choices[3]}",
        ]
        pick_label = st.radio(
            "Select your answer:",
            options=option_labels,
            index=None,  # no default selection
            key=f"pick_{qidx}",
        )
        user_pick = None if pick_label is None else pick_label.split(")", 1)[0].strip()

    with col2:
        confidence = st.radio(
            "Confidence",
            options=[1, 2, 3, 4, 5],
            index=2,
            horizontal=True,
            key=f"conf_{qidx}",
        )

    with st.form(key=f"form_{qidx}"):
        submitted = st.form_submit_button("Submit and next", disabled=(user_pick is None))

    if submitted:
        toc = time.time()
        elapsed = toc - (st.session_state.tic or toc)
        is_correct = (letter2idx(user_pick) == correct_idx) if user_pick else False

        if st.session_state.condition == "with" and qidx in st.session_state.elim_cache:
            eidxs, _rats = st.session_state.elim_cache[qidx]
            elim_letters = [idx2letter(i) for i in eidxs]
        else:
            elim_letters = []

        st.session_state.logs.append({
            "submitted_at": pd.Timestamp.utcnow().isoformat(),
            "participant_id": st.session_state.participant_id,
            "condition": st.session_state.condition,
            "trial_number": qidx + 1,
            "context": context,
            "question": question,
            "choices": json.dumps(choices, ensure_ascii=False),
            "correct_answer": idx2letter(correct_idx),
            "ai_eliminations": json.dumps(elim_letters, ensure_ascii=False),
            "user_answer": (user_pick or ""),
            "is_correct": int(is_correct),
            "time_sec": round(elapsed, 3),
            "confidence": int(confidence),
        })

        st.session_state.idx += 1
        st.session_state.tic = time.time()
        if st.session_state.idx >= FIXED_N:
            st.session_state.stage = "done"
        st.rerun()

# ----------------------------
# Sheets helpers (no fallback creation)
# ----------------------------
def _get_svc_info_as_dict():
    """Deep-convert dict-like secrets → pure dict or JSON string → dict."""
    try:
        svc_obj = st.secrets["gcp_service_account"]
        # AttrDict → dict (deep)
        def deep_to_dict(obj):
            if isinstance(obj, dict):
                return {k: deep_to_dict(v) for k, v in obj.items()}
            elif hasattr(obj, "items"):
                return {k: deep_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        if not isinstance(svc_obj, dict):
            try:
                svc_obj = dict(svc_obj)
            except Exception:
                pass
        return deep_to_dict(svc_obj)
    except Exception:
        val = st.secrets.get("gcp_service_account", None)
        if isinstance(val, str):
            return json.loads(val)
        raise

def _gs_auth():
    import gspread
    from google.oauth2.service_account import Credentials
    svc_info = _get_svc_info_as_dict()
    required = [
        "type","project_id","private_key_id","private_key","client_email",
        "client_id","auth_uri","token_uri","auth_provider_x509_cert_url","client_x509_cert_url"
    ]
    missing = [k for k in required if not svc_info.get(k)]
    if missing:
        raise RuntimeError(f"Service account JSON missing: {missing}")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(svc_info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc, svc_info

def _append_df(ws, df_session: pd.DataFrame):
    clean = df_session.where(pd.notnull(df_session), "")
    rows = [[(x.item() if hasattr(x, "item") else x) for x in r] for r in clean.to_numpy()]
    if ws.row_count < 1:
        ws.add_rows(1)
    if ws.col_count < len(df_session.columns):
        ws.add_cols(len(df_session.columns) - ws.col_count)
    if ws.acell("A1").value is None:
        ws.append_row(list(df_session.columns), value_input_option="RAW")
    ws.append_rows(rows, value_input_option="RAW")

def _normalize_sheet_id(sheet_id_or_url: str) -> str:
    if not sheet_id_or_url:
        return ""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_id_or_url)
    if m:
        return m.group(1)
    return sheet_id_or_url

def save_to_specific_sheet(df_session: pd.DataFrame, sheet_id_or_url: str):
    """Write to provided sheet only.
       Returns: ("ok", sh, svc_info) | ("perm_err", hint, svc_info) | ("err", exc, svc_info)
    """
    import gspread
    gc, svc_info = _gs_auth()
    sheet_id = _normalize_sheet_id(sheet_id_or_url)
    try:
        sh = gc.open_by_key(sheet_id)  # permission denied → raises
        try:
            ws = sh.worksheet("submissions")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="submissions", rows="1000", cols="50")
            ws.append_row(list(df_session.columns), value_input_option="RAW")
        _append_df(ws, df_session)
        return ("ok", sh, svc_info)
    except Exception as e:
        err_txt = repr(e).lower()
        if isinstance(e, PermissionError) or "403" in err_txt or "permission" in err_txt or "insufficient" in err_txt:
            hint = {
                "type": "permission",
                "client_email": svc_info.get("client_email"),
                "sheet_id": sheet_id,
                "raw": repr(e),
            }
            return ("perm_err", hint, svc_info)
        return ("err", e, svc_info)

# ----------------------------
# Done (Finish = write to provided Sheet)
# ----------------------------
if st.session_state.stage == "done":
    st.success("All trials completed. Before you leave, 1) Don't forget to Copy completion code below, return to Mturk and submit it, and then 2) make sure click the Finish button below and check the Success message. 🎉")

    if st.session_state.condition == "with" and st.session_state.ai_helpfulness_overall is None:
        st.session_state.ai_helpfulness_overall = st.radio(
            "Overall, how helpful was the AI advice?",
            options=[1, 2, 3, 4, 5],
            index=2,
            horizontal=True,
        )

    df = pd.DataFrame(st.session_state.logs)
    if st.session_state.ai_helpfulness_overall is not None:
        df["ai_helpfulness_overall"] = st.session_state.ai_helpfulness_overall

    if st.session_state.completion_code is None:
        st.session_state.completion_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    code = st.session_state.completion_code
    df["completion_code"] = code

    if not df.empty:
        acc = df["is_correct"].mean()
        total_time = df["time_sec"].sum()
        avg_time = df["time_sec"].mean()
        st.write(f"- **Accuracy**: {acc:.2%}")
        st.write(f"- **Total time**: {total_time:.1f} sec")
        st.write(f"- **Avg time / item**: {avg_time:.1f} sec")

    st.info(f"Your completion code: **{code}**")

    st.subheader("Review & Submit")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"session_{st.session_state.participant_id or 'anon'}_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv",
        mime="text/csv",
    )

    # Finish: writes to provided SHEETS_DOC_ID (no fallback)
    disabled_finish = st.session_state.finished
    if st.button("Finish", type="primary", disabled=disabled_finish):
        target_id = get_sheet_id()
        if not target_id:
            st.error("SHEETS_DOC_ID not set in secrets or env.")
        else:
            status, obj, svc = save_to_specific_sheet(df, target_id)
            if status == "ok":
                st.session_state.finished = True
                st.success("✅ Successfully done. Thank you for completing the study!")
            elif status == "perm_err":
                st.error("Permission denied when writing to the provided sheet.")
                st.write("- **Share this spreadsheet with the service account email below as an *Editor*.**")
                st.code(obj["client_email"])
                st.write("Sheet ID:")
                st.code(obj["sheet_id"])
                st.caption(f"Raw error: {obj['raw']}")
            else:
                st.error(f"Error saving data: {repr(obj)}")
