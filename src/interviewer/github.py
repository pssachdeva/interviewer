"""GitHub API integration for storing comments."""

import json
import base64
from datetime import datetime, timezone
import requests
import streamlit as st


REPO_OWNER = "pssachdeva"
REPO_NAME = "interviewer"
COMMENTS_PATH = "data/comments.jsonl"
BRANCH = "main"


def get_github_token() -> str | None:
    """Get GitHub token from Streamlit secrets."""
    try:
        return st.secrets["GITHUB_TOKEN"]
    except (KeyError, FileNotFoundError):
        return None


def load_comments() -> dict[tuple[str, int], list[dict]]:
    """Load all comments from GitHub, grouped by (transcript_id, message_index)."""
    token = get_github_token()
    if not token:
        return {}

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{COMMENTS_PATH}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.get(url, headers=headers, params={"ref": BRANCH})

    if response.status_code == 404:
        # File doesn't exist yet
        return {}

    if response.status_code != 200:
        st.error(f"Failed to load comments: {response.status_code}")
        return {}

    data = response.json()
    content = base64.b64decode(data["content"]).decode("utf-8")

    # Parse JSONL
    comments: dict[tuple[str, int], list[dict]] = {}
    for line in content.strip().split("\n"):
        if not line:
            continue
        comment = json.loads(line)
        key = (comment["transcript_id"], comment["message_index"])
        if key not in comments:
            comments[key] = []
        comments[key].append(comment)

    return comments


def save_comment(transcript_id: str, message_index: int, text: str) -> bool:
    """Save a new comment to GitHub."""
    token = get_github_token()
    if not token:
        st.error("GitHub token not configured. Add GITHUB_TOKEN to .streamlit/secrets.toml")
        return False

    # Create the new comment
    comment = {
        "transcript_id": transcript_id,
        "message_index": message_index,
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    new_line = json.dumps(comment)

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{COMMENTS_PATH}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Get current file (if exists) to get SHA
    response = requests.get(url, headers=headers, params={"ref": BRANCH})

    if response.status_code == 404:
        # File doesn't exist, create it
        new_content = new_line + "\n"
        sha = None
    elif response.status_code == 200:
        # Append to existing file
        data = response.json()
        existing_content = base64.b64decode(data["content"]).decode("utf-8")
        new_content = existing_content.rstrip("\n") + "\n" + new_line + "\n"
        sha = data["sha"]
    else:
        st.error(f"Failed to read comments file: {response.status_code}")
        return False

    # Write the updated file
    payload = {
        "message": f"Add comment on {transcript_id}:{message_index}",
        "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(url, headers=headers, json=payload)

    if response.status_code in (200, 201):
        return True
    else:
        st.error(f"Failed to save comment: {response.status_code} - {response.text}")
        return False
