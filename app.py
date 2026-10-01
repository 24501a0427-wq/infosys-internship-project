"""
AI-Powered Visual Data Analytics and Business Intelligence Platform - Streamlit dashboard (MILESTONE 4)
--------------------------------------------------
Run:  streamlit run app.py

Pages:
  🏠 Home          overview + status
  👷 PPE Detection image / video          (Milestone 1)
  📄 Documents     upload + search manuals (Milestone 2)
  🤖 Ask AI        LangGraph agent          (Milestone 3)
  📊 Dashboard     KPIs + charts            (Milestone 4)
  📋 Report        PDF download             (Milestone 4)

Remember: Streamlit re-runs this whole file on every click.
  - @st.cache_resource  -> load heavy things only once
  - st.session_state    -> remember results between clicks
  - save data only inside a button, or it is saved again on every rerun
"""
import os
import tempfile

import cv2
import streamlit as st

# Load Streamlit Cloud secret before importing modules that use it
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

import knowledge_base as kb
import storage
import vision
from agent import ask_agent
from documents import process_file
from llm import MODEL, has_key
from report import build_pdf, write_summary

st.set_page_config(page_title="AI-Powered Visual Data Analytics and Business Intelligence Platform", page_icon="🦺", layout="wide")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#060a10;--panel:#0d141e;--panel2:#111b27;--line:#243244;--text:#f4f7fb;--muted:#8e9aaa;--yellow:#ffd21c;--cyan:#59e1ef;--green:#4ade80;--red:#ff5d73}
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:radial-gradient(circle at 85% 5%,rgba(255,210,28,.08),transparent 25%),radial-gradient(circle at 0 80%,rgba(89,225,239,.04),transparent 22%),var(--bg);color:var(--text)}
.block-container{max-width:1450px;padding-top:2rem;padding-bottom:3rem}section[data-testid="stSidebar"]{background:#080d14;border-right:1px solid var(--line)}
.brand{padding:8px 5px 20px}.mark{width:44px;height:44px;border:1px solid rgba(255,210,28,.55);border-radius:13px;display:flex;align-items:center;justify-content:center;color:var(--yellow);font-size:24px;box-shadow:0 0 25px rgba(255,210,28,.1)}.brand h2{font:700 22px 'Space Grotesk';margin:12px 0 2px}.brand p{font-size:10px;color:var(--muted);letter-spacing:1.8px;text-transform:uppercase}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:4px}section[data-testid="stSidebar"] div[role="radiogroup"] label{border:1px solid transparent;border-radius:11px;padding:8px 9px}section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:#101924;border-color:#1e2b3a}
.status{background:linear-gradient(145deg,#101925,#0a1018);border:1px solid var(--line);border-radius:14px;padding:13px;margin-top:18px}.sr{display:flex;justify-content:space-between;padding:7px 0;font-size:11px;color:#b7c1cd}.dot{width:7px;height:7px;border-radius:50%;display:inline-block;background:var(--green);margin-right:7px;box-shadow:0 0 9px rgba(74,222,128,.5)}
.hero{position:relative;overflow:hidden;background:linear-gradient(135deg,rgba(255,210,28,.1),transparent 42%),linear-gradient(145deg,#111b28,#090f17);border:1px solid #29394c;border-radius:24px;padding:42px 46px;margin-bottom:25px;box-shadow:0 20px 65px rgba(0,0,0,.3)}.hero:after{content:"";position:absolute;width:260px;height:260px;border:1px solid rgba(255,210,28,.14);border-radius:50%;right:-80px;top:-105px;box-shadow:0 0 0 30px rgba(255,210,28,.025),0 0 0 60px rgba(255,210,28,.018)}.eyebrow,.kicker{color:var(--yellow);font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase}.hero h1{font:700 clamp(40px,5vw,68px)/.98 'Space Grotesk';letter-spacing:-2.5px;margin:10px 0}.hero h1 span{color:var(--yellow)}.hero p{max-width:780px;color:#aab5c3;line-height:1.7;font-size:15px}.page-title{font:700 36px 'Space Grotesk';letter-spacing:-1px;margin:5px 0}.page-desc{color:var(--muted);margin-bottom:24px}
.card{background:linear-gradient(145deg,rgba(17,27,39,.96),rgba(9,15,23,.96));border:1px solid var(--line);border-radius:17px;padding:21px;height:100%;box-sizing:border-box}.accent{border-top:2px solid var(--yellow)}.ct{font:700 17px 'Space Grotesk';margin-bottom:7px}.muted{color:var(--muted);font-size:13px;line-height:1.65}.ibox{width:42px;height:42px;border-radius:12px;background:rgba(255,210,28,.08);border:1px solid rgba(255,210,28,.18);display:flex;align-items:center;justify-content:center;color:var(--yellow);font-size:19px;margin-bottom:14px}
.pipeline{display:flex;align-items:center;gap:8px;overflow:auto;margin:15px 0 28px}.node{min-width:145px;background:#0d151f;border:1px solid var(--line);border-radius:13px;padding:13px;text-align:center}.node b{display:block;font-size:11px}.node small{color:var(--muted);font-size:10px}.arrow{color:var(--yellow);font-weight:800}
.line{display:flex;align-items:center;gap:12px;margin:28px 0 15px}.line span{color:var(--yellow);font-size:10px;font-weight:800;letter-spacing:1.7px;text-transform:uppercase}.line:after{content:"";height:1px;background:var(--line);flex:1}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#111b27,#0b121b);border:1px solid var(--line);padding:16px;border-radius:15px}div[data-testid="stMetricLabel"]{color:var(--muted)}div[data-testid="stMetricValue"]{font-family:'Space Grotesk';color:var(--text)}
.stButton>button,.stDownloadButton>button{border-radius:10px;min-height:42px;font-weight:700;border:1px solid #354355;background:#111a25;color:var(--text)}.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--yellow);color:var(--yellow)}.stButton>button[kind="primary"]{background:var(--yellow);color:#070a0e;border-color:var(--yellow)}
.stTextInput input,.stTextArea textarea,.stSelectbox [data-baseweb="select"]>div,.stFileUploader section{background:#0d141e!important;border-color:#273548!important;border-radius:10px!important}.stFileUploader section{border:1px dashed #39495c!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}.footer{margin-top:48px;border-top:1px solid var(--line);padding-top:15px;color:#667384;font-size:10px;display:flex;justify-content:space-between}
</style>""", unsafe_allow_html=True)





# ── load once ──────────────────────────────────────────────
@st.cache_resource
def get_model():
    return vision.load_model()


@st.cache_resource
def load_sample_manual():
    """If the knowledge base is empty (e.g. fresh cloud start), add a manual automatically."""
    if kb.total_chunks() == 0:
        for name in ["safety_manual.pdf", "sample_safety_manual.txt"]:
            path = os.path.join(vision.BASE_DIR, "data", name)
            if os.path.exists(path):
                kb.add_document(process_file(path), name)
                return name
    return None


def save_upload(uploaded):
    suffix = os.path.splitext(uploaded.name)[1]          # keep the real extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


model = get_model()
load_sample_manual()

# ── sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div class="brand"><div class="mark">◈</div><h2>VISIONDESK AI</h2><p>JARVIS · Safety Intelligence</p></div>""", unsafe_allow_html=True)
    page = st.radio("Go to", ["🏠 Home", "👷 PPE Detection", "📄 Documents",
                              "🤖 Ask AI", "📊 Dashboard", "📋 Report"])
    st.divider()
    st.markdown(f"""<div class="status"><div class="sr"><span><i class="dot"></i>PPE model</span><b>{'ONLINE' if model else 'OFFLINE'}</b></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="sr"><span><i class="dot"></i>Gemini</span><b>{'READY' if has_key() else 'NO KEY'}</b></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="sr"><span><i class="dot"></i>Knowledge base</span><b>{kb.total_chunks()} CHUNKS</b></div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 🏠 HOME
# ══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""<div class="hero"><div class="eyebrow">JARVIS · WORKPLACE INTELLIGENCE</div><h1>See the site.<br><span>Understand the risk.</span></h1><p>VisionDesk AI combines computer vision, document intelligence, grounded AI reasoning and business analytics into one safety intelligence workspace.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="pipeline"><div class="node"><b>01 · VISION</b><small>PPE detection</small></div><div class="arrow">→</div><div class="node"><b>02 · KNOWLEDGE</b><small>Safety documents</small></div><div class="arrow">→</div><div class="node"><b>03 · REASONING</b><small>Gemini + LangGraph</small></div><div class="arrow">→</div><div class="node"><b>04 · ANALYTICS</b><small>KPIs + reports</small></div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="line"><span>System Overview</span></div>""", unsafe_allow_html=True)
    st.markdown("""
1. **👷 PPE Detection** – upload a site image or video, see violations
2. **📄 Documents** – upload safety manuals and inspection reports
3. **🤖 Ask AI** – ask questions; the AI uses the manual **and** the last image result
4. **📊 Dashboard** – compliance rate, violations, trends
5. **📋 Report** – download a PDF compliance report
""")
    if not model:
        st.warning("Put your trained **best.pt** in the project folder to enable detection.")
    if not has_key():
        st.info("Add **GEMINI_API_KEY** to the `.env` file for AI answers.")


# ══════════════════════════════════════════════════════════
# 👷 PPE DETECTION  (Milestone 1)
# ══════════════════════════════════════════════════════════
elif page == "👷 PPE Detection":
    st.markdown('<div class="kicker">MILESTONE 01 · VISION</div><div class="page-title">PPE Detection</div><div class="page-desc">Computer vision inspection for workplace safety evidence.</div>', unsafe_allow_html=True)
    if not model:
        st.error("best.pt not found in the project folder.")
        st.stop()

    kind = st.radio("Input", ["Image", "Video"], horizontal=True)
    conf = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05,
                     help="Higher = fewer false alarms, but may miss some items")

    if kind == "Image":
        file = st.file_uploader("Upload a site image", type=["jpg", "jpeg", "png"])
        if file and st.button("Analyze image", type="primary"):
            path = save_upload(file)
            dets = vision.detect(model, path, conf)
            result = vision.summarize(dets, file.name)
            storage.log_scan(file.name, result, "image")           # saved only on click
            st.session_state["last"] = {"name": file.name, "result": result, "dets": dets,
                                        "image": vision.draw(cv2.imread(path), dets)}
    else:
        file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        every_n = st.slider("Check 1 of every N frames", 1, 30, 5)
        if file and st.button("Analyze video", type="primary"):
            with st.spinner("Analysing video..."):
                dets, preview, frames = vision.detect_video(model, save_upload(file), every_n, conf)
            result = vision.summarize(dets, file.name)
            storage.log_scan(file.name, result, "video")
            st.session_state["last"] = {"name": f"{file.name} ({frames} frames)", "result": result,
                                        "dets": dets, "image": preview}

    last = st.session_state.get("last")
    if last:
        col1, col2 = st.columns([3, 2])
        if last["image"] is not None:
            col1.image(cv2.cvtColor(last["image"], cv2.COLOR_BGR2RGB), caption=last["name"])
        with col2:
            r = last["result"]
            if r["compliant"]:
                st.success("✅ COMPLIANT - no violations")
            else:
                st.error("🚨 VIOLATIONS: " + ", ".join(sorted(set(r["violations"]))))
            st.metric("Objects detected", len(last["dets"]))
            st.metric("Violations", len(r["violations"]))
            st.write("**Counts:**", r["counts"])
            st.caption("This result is now available in 🤖 Ask AI.")


# ══════════════════════════════════════════════════════════
# 📄 DOCUMENTS  (Milestone 2)
# ══════════════════════════════════════════════════════════
elif page == "📄 Documents":
    st.markdown('<div class="kicker">MILESTONE 02 · KNOWLEDGE</div><div class="page-title">Safety Knowledge Base</div><div class="page-desc">Transform unstructured manuals and reports into searchable semantic knowledge.</div>', unsafe_allow_html=True)
    files = st.file_uploader("Upload manuals / inspection or incident reports",
                             type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if files and st.button("Add to knowledge base", type="primary"):
        for f in files:
            chunks = process_file(save_upload(f))
            if chunks:
                kb.add_document(chunks, f.name)
                st.success(f"{f.name}: {len(chunks)} chunks added")
            else:
                st.warning(f"{f.name}: no text found (scanned PDF?)")

    st.subheader("Documents in the knowledge base")
    docs = kb.list_documents()
    if docs:
        st.table([{"document": name, "chunks": n} for name, n in docs.items()])
    else:
        st.info("No documents yet.")

    st.subheader("🔍 Test search")
    q = st.text_input("Search", placeholder="When must goggles be worn?")
    if q:
        for r in kb.search(q):
            st.info(f"**{r['source']} #{r['chunk']}** (distance {r['distance']})\n\n{r['text']}")


# ══════════════════════════════════════════════════════════
# 🤖 ASK AI  (Milestone 3 - LangGraph agent)
# ══════════════════════════════════════════════════════════
elif page == "🤖 Ask AI":
    st.markdown('<div class="kicker">MILESTONE 03 · REASONING</div><div class="page-title">Safety AI</div><div class="page-desc">Ask questions using visual observations and documented safety rules.</div>', unsafe_allow_html=True)

    last = st.session_state.get("last")
    use_image = False
    if last:
        use_image = st.checkbox(f"Include last detection result ({last['name']})", value=True)
        if use_image:
            st.caption(last["result"]["text"])

    question = st.text_input("Your question", placeholder="Is this worker compliant? Which rule applies?")

    if st.button("Ask", type="primary") and question:
        vis = last["result"] if (use_image and last) else None
        with st.spinner("Thinking..."):
            out = ask_agent(question,
                            vision=vis["text"] if vis else "",
                            violations=vis["violations"] if vis else [])
        st.session_state["answer"] = {"question": question, **out}
        st.session_state["voted"] = False

    ans = st.session_state.get("answer")
    if ans:
        st.markdown("### Answer")
        st.write(ans["answer"])
        with st.expander("How the agent worked"):
            st.write("**Steps:** " + " → ".join(ans["steps"]))
            for d in ans["docs"]:
                st.caption(f"[{d['source']} #{d['chunk']}] {d['text'][:200]}")

        if not st.session_state.get("voted"):
            st.write("Was this answer helpful?")
            c1, c2 = st.columns(2)
            if c1.button("👍 Yes"):
                storage.log_feedback(ans["question"], True)
                st.session_state["voted"] = True
                st.rerun()
            if c2.button("👎 No"):
                storage.log_feedback(ans["question"], False)
                st.session_state["voted"] = True
                st.rerun()
        else:
            st.caption("Thanks for your feedback!")


# ══════════════════════════════════════════════════════════
# 📊 DASHBOARD  (Milestone 4)
# ══════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown('<div class="kicker">MILESTONE 04 · ANALYZE</div><div class="page-title">Safety Analytics</div><div class="page-desc">Turn inspection history into operational KPIs and trend intelligence.</div>', unsafe_allow_html=True)
    scans = storage.load_scans()
    k = storage.kpis(scans)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images / videos checked", k["scans"])
    c2.metric("Compliance rate", f"{k['compliance']}%" if k["compliance"] is not None else "-")
    c3.metric("Total violations", k["violations"])
    c4.metric("User satisfaction", f"{k['satisfaction']}%" if k["satisfaction"] is not None else "-",
              help="Target: 85% or more")

    if scans.empty:
        st.info("No data yet. Analyse some images in 👷 PPE Detection.")
    else:
        left, right = st.columns(2)
        left.subheader("Violations by type")
        counts = storage.violation_counts(scans)
        if len(counts):
            left.bar_chart(counts)
        else:
            left.success("No violations so far 🎉")
        right.subheader("Violations per day")
        right.line_chart(scans.set_index("time")["violations"].resample("D").sum())

        st.subheader("All scans")
        st.dataframe(scans.sort_values("time", ascending=False), hide_index=True)

    if st.button("🗑️ Clear history"):
        storage.clear_all()
        st.rerun()


# ══════════════════════════════════════════════════════════
# 📋 REPORT  (Milestone 4)
# ══════════════════════════════════════════════════════════
elif page == "📋 Report":
    st.markdown('<div class="kicker">MILESTONE 04 · REPORT</div><div class="page-title">Compliance Reports</div><div class="page-desc">Generate an AI-assisted summary and export the inspection history.</div>', unsafe_allow_html=True)
    scans = storage.load_scans()

    if st.button("Generate report", type="primary"):
        k = storage.kpis(scans)
        counts = storage.violation_counts(scans)
        with st.spinner("Writing summary..."):
            summary = write_summary(k, counts)
        st.session_state["report"] = {"summary": summary, "pdf": build_pdf(k, counts, scans, summary)}

    rep = st.session_state.get("report")
    if rep:
        st.subheader("Summary")
        st.write(rep["summary"])
        st.download_button("⬇️ Download PDF", rep["pdf"],
                           file_name="safety_compliance_report.pdf", mime="application/pdf")
        st.download_button("⬇️ Download data (CSV)", scans.to_csv(index=False),
                           file_name="scans.csv", mime="text/csv")
