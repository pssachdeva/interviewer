"""Mobile-friendly interview viewer dashboard."""

import streamlit as st
from datasets import load_dataset

import sys
sys.path.insert(0, "src")
from interviewer.parser import parse_transcript


st.set_page_config(
    page_title="Anthropic Interviews",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Mobile-friendly dark mode CSS
st.markdown("""
<style>
    /* Dark mode base */
    .stApp {
        background-color: #1a1a1a;
    }

    /* Hide hamburger menu and footer for cleaner mobile view */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Constrain width for desktop, center content */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 500px !important;
        margin: 0 auto;
    }

    /* Chat bubble base styles */
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 85%;
        word-wrap: break-word;
        line-height: 1.4;
        font-size: 15px;
    }

    /* Assistant bubbles - left aligned, dark gray */
    .assistant-bubble {
        background-color: #2a2a2a;
        color: #e0e0e0;
        margin-right: auto;
        margin-left: 0;
        border-bottom-left-radius: 4px;
    }

    /* User bubbles - right aligned, blue */
    .user-bubble {
        background-color: #0084ff;
        color: white;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 4px;
    }

    /* Container for proper alignment */
    .bubble-container {
        display: flex;
        width: 100%;
    }

    .bubble-container.user {
        justify-content: flex-end;
    }

    .bubble-container.assistant {
        justify-content: flex-start;
    }

    /* Add padding at bottom so content isn't hidden by nav */
    .main-content {
        padding-bottom: 80px;
    }

    /* Header styling */
    .interview-header {
        font-size: 14px;
        color: #888;
        text-align: center;
        padding: 8px 0;
        border-bottom: 1px solid #333;
        margin-bottom: 16px;
        position: sticky;
        top: 0;
        background: #1a1a1a;
        z-index: 100;
    }

    /* Split selector - dark mode */
    .stSelectbox {
        max-width: 200px;
        margin: 0 auto;
    }

    .stSelectbox > div > div {
        background-color: #2a2a2a;
        color: #e0e0e0;
    }

    /* Make buttons larger for touch */
    .stButton > button {
        min-height: 44px;
        min-width: 44px;
        font-size: 18px;
        background-color: #2a2a2a;
        color: #e0e0e0;
        border: 1px solid #444;
    }

    .stButton > button:hover {
        background-color: #3a3a3a;
        border-color: #555;
    }

    /* Number input dark mode */
    .stNumberInput > div > div > input {
        background-color: #2a2a2a;
        color: #e0e0e0;
        border-color: #444;
    }

    /* Divider */
    hr {
        border-color: #333;
    }

    /* Reduce default Streamlit element gaps */
    .stButton {
        margin-bottom: 0 !important;
    }

    .element-container {
        margin-bottom: 0 !important;
    }

    /* Remove gap after columns (nav buttons) */
    .stColumns {
        margin-bottom: 0 !important;
        gap: 0.5rem !important;
    }

    /* Target the horizontal block that wraps columns */
    [data-testid="stHorizontalBlock"] {
        margin-bottom: 0 !important;
        gap: 0.5rem !important;
    }

    /* Reduce vertical spacing on all elements */
    .stMarkdown {
        margin-bottom: 0 !important;
    }

    div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_interviews():
    """Load and parse all interviews from the dataset."""
    ds = load_dataset("Anthropic/AnthropicInterviewer")

    interviews = []
    for split in ["workforce", "creatives", "scientists"]:
        for row in ds[split]:
            interviews.append({
                "id": row["transcript_id"],
                "split": split,
                "text": row["text"],
                "messages": parse_transcript(row["text"]),
            })

    return interviews


# Load data
interviews = load_all_interviews()

# Initialize session state
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "selected_split" not in st.session_state:
    st.session_state.selected_split = "all"


# Filter by split
split_options = ["all", "workforce", "creatives", "scientists"]
selected_split = st.selectbox(
    "Filter by group",
    split_options,
    index=split_options.index(st.session_state.selected_split),
    key="split_selector",
)

if selected_split != st.session_state.selected_split:
    st.session_state.selected_split = selected_split
    st.session_state.current_index = 0

if selected_split == "all":
    filtered_interviews = interviews
else:
    filtered_interviews = [i for i in interviews if i["split"] == selected_split]

total_count = len(filtered_interviews)
current_index = st.session_state.current_index

# Ensure index is valid
if current_index >= total_count:
    current_index = 0
    st.session_state.current_index = 0

current_interview = filtered_interviews[current_index]

# Header
st.markdown(
    f'<div class="interview-header">'
    f'<strong>{current_interview["id"]}</strong> · {current_interview["split"]} · '
    f'{current_index + 1} of {total_count}'
    f'</div>',
    unsafe_allow_html=True
)

# Top navigation
top_col1, top_col2, top_col3 = st.columns([1, 2, 1])
with top_col1:
    if st.button("← Prev", key="prev_top", use_container_width=True, disabled=(current_index == 0)):
        st.session_state.current_index = current_index - 1
        st.rerun()
with top_col3:
    if st.button("Next →", key="next_top", use_container_width=True, disabled=(current_index >= total_count - 1)):
        st.session_state.current_index = current_index + 1
        st.rerun()

# Chat messages
st.markdown('<div class="main-content">', unsafe_allow_html=True)

for msg in current_interview["messages"]:
    bubble_class = "assistant-bubble" if msg.role == "assistant" else "user-bubble"
    container_class = msg.role

    # Escape HTML in content and convert newlines
    content = msg.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    content = content.replace("\n", "<br>")

    st.markdown(
        f'<div class="bubble-container {container_class}">'
        f'<div class="chat-bubble {bubble_class}">{content}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Navigation buttons at bottom
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("← Prev", key="prev_bottom", use_container_width=True, disabled=(current_index == 0)):
        st.session_state.current_index = current_index - 1
        st.rerun()

with col2:
    # Jump to specific interview
    new_index = st.number_input(
        "Go to",
        min_value=1,
        max_value=total_count,
        value=current_index + 1,
        label_visibility="collapsed",
    )
    if new_index - 1 != current_index:
        st.session_state.current_index = new_index - 1
        st.rerun()

with col3:
    if st.button("Next →", key="next_bottom", use_container_width=True, disabled=(current_index >= total_count - 1)):
        st.session_state.current_index = current_index + 1
        st.rerun()
