import streamlit as st
import html
from pathlib import Path

from src.retrieve import Retriever
from src.generate import generate_answer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediQuery | Clinical Knowledge Assistant",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent
HERO_IMAGE = PROJECT_DIR / "assets" / "med1.jpeg"


# ============================================================
# 2. CUSTOM DESIGN SYSTEM
# ============================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --background: #0B1220;
    --sidebar: #111C2E;
    --card: #142238;
    --border: #263B55;
    --primary: #36D6C3;
    --text: #EDF4FC;
    --muted: #94A7BD;
}

/* GLOBAL TYPOGRAPHY */

html, body, [class*="css"],
[data-testid="stApp"],
.stMarkdown,
.stText,
label,
input,
textarea,
button {
    font-family: 'Manrope', sans-serif !important;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    font-family: 'Manrope', sans-serif !important;
}

section[data-testid="stSidebar"] hr {
    border-color: var(--border);
}

/* BRAND */

.brand {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: var(--text);
}

.brand span {
    color: var(--primary);
}

.brand-caption {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--muted);
    margin-top: 5px;
}

/* HERO */

.hero {
    padding: 30px 0 25px 0;
}

.hero-label {
    color: var(--primary);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.5px;
    margin-bottom: 18px;
}

.hero-title {
    color: var(--text);
    font-size: 43px;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.2;
}

.hero-title span {
    color: var(--primary);
}

.hero-subtitle {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.9;
    margin-top: 18px;
    max-width: 560px;
}

/* HERO IMAGE */

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid var(--border);
}

/* STATUS BADGE */

.status-badge {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 30px;
    background: rgba(54,214,195,0.10);
    color: var(--primary);
    border: 1px solid rgba(54,214,195,0.25);
    font-size: 11px;
    font-weight: 700;
}

/* INFORMATION CARDS */

.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 22px;
    min-height: 125px;
}

.info-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.info-value {
    color: var(--text);
    font-size: 21px;
    font-weight: 800;
    margin-top: 15px;
}

/* SECTION HEADINGS */

.section-heading {
    font-size: 23px;
    font-weight: 800;
    color: var(--text);
    margin-top: 25px;
}

.section-description {
    color: var(--muted);
    font-size: 13px;
    margin-top: 7px;
    margin-bottom: 20px;
}

/* SUGGESTED QUESTION BUTTONS */

div[data-testid="stButton"] button {
    background: #183448 !important;
    color: #DCECF5 !important;
    border: 1px solid #31566A !important;
    border-radius: 11px !important;
    min-height: 65px;
    padding: 12px 15px;
    transition: all 0.2s ease;
}

div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span,
div[data-testid="stButton"] button div {
    font-family: 'Manrope', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #DCECF5 !important;
}

div[data-testid="stButton"] button:hover {
    background: #24536A !important;
    border-color: var(--primary) !important;
}

/* CHAT INPUT */

.stChatInput textarea {
    font-family: 'Manrope', sans-serif !important;
    font-size: 14px !important;
}

.stChatInput > div {
    background: var(--card) !important;
    border: 1px solid #344B67 !important;
    border-radius: 13px !important;
}

/* CHAT MESSAGES */

[data-testid="stChatMessage"] {
    border-radius: 13px;
}

[data-testid="stChatMessage"] p {
    font-family: 'Manrope', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.8;
}

/* SOURCE CARDS */

.source-card {
    background: #111C2E;
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 10px;
    padding: 17px;
    margin: 10px 0;
}

.source-title {
    color: var(--primary);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.source-meta {
    color: #93A8C0;
    font-size: 11px;
    margin-top: 7px;
}

.source-text {
    color: #D5E1EF;
    font-size: 13px;
    line-height: 1.8;
    margin-top: 13px;
}

/* EXPANDERS */

[data-testid="stExpander"] {
    background: #111C2E;
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* DIVIDERS */

hr {
    border-color: var(--border);
}

/* DISCLAIMER */

.disclaimer {
    text-align: center;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.9;
    padding: 20px;
}

.disclaimer strong {
    color: var(--primary);
}

</style>
""")


# ============================================================
# 3. LOAD RETRIEVAL ENGINE
# ============================================================

@st.cache_resource
def load_retriever():
    return Retriever()


try:
    with st.spinner("Connecting to the knowledge base..."):
        retriever = load_retriever()

except Exception as e:
    st.error("Unable to connect to the knowledge base.")
    st.caption(f"Technical details: {e}")
    st.stop()


# ============================================================
# 4. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def display_sources(sources):

    if not sources:
        return

    with st.expander(
        f"Explore retrieved evidence ({len(sources)} passages)"
    ):

        for i, source in enumerate(sources, 1):

            document = html.escape(str(source["document"]))
            page = html.escape(str(source["page"]))
            passage = html.escape(str(source["text"]))

            st.html(f"""
            <div class="source-card">

                <div class="source-title">
                    EVIDENCE SOURCE {i}
                </div>

                <div class="source-meta">
                    {document} | PDF page {page}
                </div>

                <div class="source-text">
                    {passage}
                </div>

            </div>
            """)


def process_question(question):

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Searching clinical documents..."):

                passages = retriever.search(
                    question,
                    top_k=6
                )

            with st.spinner("Generating your answer..."):

                answer = generate_answer(
                    question,
                    passages
                )

            st.markdown(answer)

            display_sources(passages)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": passages
            })

        except Exception as e:

            st.error(
                "We couldn't complete your request. "
                "Please try again shortly."
            )

            # Keep technical details in the terminal,
            # rather than exposing a traceback in the interface.
            print(f"Generation error: {repr(e)}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "The request could not be completed. "
                    "Please try again."
                ),
                "sources": []
            })


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.html("""
    <div style="padding: 15px 0 25px 0;">

        <div class="brand">
            Medi<span>Query</span>
        </div>

        <div class="brand-caption">
            CLINICAL KNOWLEDGE ASSISTANT
        </div>

    </div>
    """)

    st.divider()

    st.markdown("### Workspace")

    st.html("""
    <div class="status-badge">
        ● KNOWLEDGE BASE CONNECTED
    </div>
    """)

    st.write("")

    if st.button(
        "＋ New conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### Knowledge base")

    st.markdown("**Nigeria Essential Medicines List**")

    st.caption("""
    8th Edition · 2024

    Federal Ministry of Health and Social Welfare
    """)

    st.markdown("""
    <small style="color:#94A7BD;">
    Indexed document: Adult Essential Medicines List<br>
    Indexed passages: 193 text chunks<br>
    Retrieval: Dense + BM25 hybrid search
    </small>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### About MediQuery")

    st.caption(
        "An evidence-grounded health information assistant "
        "that retrieves relevant information from official "
        "Nigerian health documents."
    )

    st.divider()

    st.caption("Prototype · Version 0.3")


# ============================================================
# 7. HERO SECTION
# ============================================================

if not st.session_state.messages:

    hero_left, hero_right = st.columns(
        [1.4, 1],
        vertical_alignment="center"
    )

    with hero_left:

        st.html("""
        <div class="hero">

            <div class="hero-label">
                YOUR CLINICAL KNOWLEDGE WORKSPACE
            </div>

            <div class="hero-title">
                Find evidence.<br>
                <span>Understand better.</span>
            </div>

            <div class="hero-subtitle">
                Explore Nigerian health documents through
                natural-language questions. Retrieve relevant
                evidence and receive answers grounded in
                their original sources.
            </div>

        </div>
        """)

    with hero_right:

        if HERO_IMAGE.exists():

            st.image(
                str(HERO_IMAGE),
                use_container_width=True
            )

        else:

            st.warning(
                "Hero image not found. Place med1.jpeg "
                "inside the assets folder."
            )

else:

    st.html("""
    <div style="padding:15px 0 20px 0;">

        <div class="hero-label">
            MEDIQUERY WORKSPACE
        </div>

        <div style="
            font-size:28px;
            font-weight:800;
            color:#EDF4FC;
        ">
            Clinical knowledge, within reach.
        </div>

        <div style="
            color:#94A7BD;
            font-size:13px;
            margin-top:8px;
        ">
            Ask another question or explore your conversation.
        </div>

    </div>
    """)


# ============================================================
# 8. KNOWLEDGE BASE SUMMARY
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.html("""
    <div class="info-card">
        <div class="info-label">KNOWLEDGE SOURCE</div>
        <div class="info-value">NEML 2024</div>
    </div>
    """)

with col2:

    st.html("""
    <div class="info-card">
        <div class="info-label">INDEXED PASSAGES</div>
        <div class="info-value">193 Chunks</div>
    </div>
    """)

with col3:

    st.html("""
    <div class="info-card">
        <div class="info-label">RETRIEVAL ENGINE</div>
        <div class="info-value">Hybrid Search</div>
    </div>
    """)


st.write("")
st.divider()


# ============================================================
# 9. SUGGESTED QUESTIONS
# ============================================================

if not st.session_state.messages:

    st.html("""
    <div class="section-heading">
        Start exploring
    </div>

    <div class="section-description">
        Select a suggested question or enter your own.
    </div>
    """)

    suggestions = [
        "Which antihypertensive medicines are included in the NEML?",
        "What formulations of amoxicillin are listed?",
        "Which antimalarial medicines are included?"
    ]

    suggestion_cols = st.columns(3)

    for i, suggestion in enumerate(suggestions):

        with suggestion_cols[i]:

            if st.button(
                suggestion,
                key=f"suggestion_{i}",
                use_container_width=True
            ):

                st.session_state.pending_question = suggestion
                st.rerun()


# ============================================================
# 10. CONVERSATION HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.markdown(message["content"])

    else:

        with st.chat_message("assistant"):

            st.markdown(message["content"])

            display_sources(
                message.get("sources", [])
            )


# ============================================================
# 11. QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the loaded health documents..."
)

if st.session_state.pending_question:

    question = st.session_state.pending_question
    st.session_state.pending_question = None


# ============================================================
# 12. GENERATE ANSWER
# ============================================================

if question:

    process_question(question)


# ============================================================
# 13. FOOTER
# ============================================================

st.write("")
st.write("")

st.divider()

st.html("""
<div class="disclaimer">

    <strong>MEDIQUERY</strong>
    &nbsp; · &nbsp;
    Evidence-grounded health information

    <br>

    This prototype retrieves information from its loaded
    documents. It is not a substitute for professional
    clinical judgement or medical advice.

    <br>

    Always consult the original source document when
    verifying clinical information.

</div>
""")