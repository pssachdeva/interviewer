"""Mobile-friendly interview viewer dashboard."""

import streamlit as st
from datasets import load_dataset

import sys
sys.path.insert(0, "src")
from interviewer.parser import parse_transcript
from interviewer.github import load_comments, save_comment, get_github_token


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

    /* User bubble with actions row */
    .user-bubble-wrapper {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        max-width: 85%;
        margin-left: auto;
    }

    .bubble-actions {
        display: flex;
        gap: 8px;
        margin-top: 4px;
        align-items: center;
    }

    .action-btn {
        background: #333;
        border: none;
        color: #888;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        cursor: pointer;
        font-size: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .action-btn:hover {
        background: #444;
        color: #fff;
    }

    .comment-count {
        background: #ff6b35;
        color: white;
        min-width: 24px;
        height: 24px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        padding: 0 6px;
    }

    .comment-count:hover {
        background: #ff8555;
    }

    /* Comment bubbles - distinct style */
    .comment-bubble {
        background-color: #3d2a1a;
        color: #f0d0a0;
        padding: 10px 14px;
        border-radius: 12px;
        margin: 4px 0;
        font-size: 14px;
        border-left: 3px solid #ff6b35;
    }

    .comment-timestamp {
        font-size: 11px;
        color: #888;
        margin-top: 4px;
    }

    .comments-section {
        margin: 8px 0 16px 0;
        padding-left: 20px;
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

    /* Small inline buttons for comment actions */
    .small-btn button {
        min-height: 32px !important;
        min-width: 32px !important;
        font-size: 14px !important;
        padding: 0 12px !important;
    }

    /* Number input dark mode */
    .stNumberInput > div > div > input {
        background-color: #2a2a2a;
        color: #e0e0e0;
        border-color: #444;
    }

    /* Text area dark mode */
    .stTextArea textarea {
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


@st.cache_data(ttl=60)
def load_comments_cached():
    """Load comments with short TTL for freshness."""
    return load_comments()


# Load data
interviews = load_all_interviews()
all_comments = load_comments_cached()

# Initialize session state
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "selected_split" not in st.session_state:
    st.session_state.selected_split = "all"
if "adding_comment_to" not in st.session_state:
    st.session_state.adding_comment_to = None  # (transcript_id, message_index) or None
if "expanded_comments" not in st.session_state:
    st.session_state.expanded_comments = set()  # set of (transcript_id, message_index)


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
transcript_id = current_interview["id"]

# Header
st.markdown(
    f'<div class="interview-header">'
    f'<strong>{transcript_id}</strong> · {current_interview["split"]} · '
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

has_github_token = get_github_token() is not None

for msg_idx, msg in enumerate(current_interview["messages"]):
    # Escape HTML in content and convert newlines
    content = msg.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    content = content.replace("\n", "<br>")

    if msg.role == "assistant":
        # Assistant message - simple bubble
        st.markdown(
            f'<div class="bubble-container assistant">'
            f'<div class="chat-bubble assistant-bubble">{content}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        # User message - bubble with comment actions
        comment_key = (transcript_id, msg_idx)
        msg_comments = all_comments.get(comment_key, [])
        comment_count = len(msg_comments)
        is_expanded = comment_key in st.session_state.expanded_comments
        is_adding = st.session_state.adding_comment_to == comment_key

        # Render the user bubble
        st.markdown(
            f'<div class="bubble-container user">'
            f'<div class="chat-bubble user-bubble">{content}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Action buttons row (only if GitHub token configured)
        if has_github_token:
            btn_cols = st.columns([3, 1, 1] if comment_count > 0 else [4, 1])

            with btn_cols[-1]:
                # Add comment button
                if st.button("＋", key=f"add_{msg_idx}", help="Add comment"):
                    if is_adding:
                        st.session_state.adding_comment_to = None
                    else:
                        st.session_state.adding_comment_to = comment_key
                    st.rerun()

            if comment_count > 0:
                with btn_cols[-2]:
                    # Comment count badge
                    if st.button(f"{comment_count}", key=f"count_{msg_idx}", help="Show/hide comments"):
                        if is_expanded:
                            st.session_state.expanded_comments.discard(comment_key)
                        else:
                            st.session_state.expanded_comments.add(comment_key)
                        st.rerun()

        # Show existing comments if expanded
        if is_expanded and msg_comments:
            st.markdown('<div class="comments-section">', unsafe_allow_html=True)
            for comment in msg_comments:
                comment_text = comment["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                comment_text = comment_text.replace("\n", "<br>")
                timestamp = comment.get("timestamp", "")[:10]  # Just the date
                st.markdown(
                    f'<div class="comment-bubble">{comment_text}'
                    f'<div class="comment-timestamp">{timestamp}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # Show add comment form if active
        if is_adding:
            new_comment = st.text_area(
                "Add comment",
                key=f"comment_text_{msg_idx}",
                height=80,
                label_visibility="collapsed",
                placeholder="Write your comment...",
            )
            submit_col, cancel_col = st.columns(2)
            with submit_col:
                if st.button("Submit", key=f"submit_{msg_idx}", use_container_width=True):
                    if new_comment.strip():
                        if save_comment(transcript_id, msg_idx, new_comment.strip()):
                            st.session_state.adding_comment_to = None
                            # Clear the cache to reload comments
                            load_comments_cached.clear()
                            st.success("Comment saved!")
                            st.rerun()
            with cancel_col:
                if st.button("Cancel", key=f"cancel_{msg_idx}", use_container_width=True):
                    st.session_state.adding_comment_to = None
                    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Show warning if no GitHub token
if not has_github_token:
    st.caption("💡 Add GITHUB_TOKEN to secrets to enable comments")

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
