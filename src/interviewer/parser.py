"""Parse interview transcripts into structured messages."""

import re
from dataclasses import dataclass


@dataclass
class Message:
    role: str  # 'assistant' or 'user'
    content: str


ASSISTANT_MARKERS = ('A:', 'AI:', 'Assistant:')
USER_MARKERS = ('User:',)
ALL_MARKERS = ASSISTANT_MARKERS + USER_MARKERS


def parse_transcript(text: str) -> list[Message]:
    """Parse a transcript into a list of messages.

    Handles formats like:
    - "A: ..." / "AI: ..." / "Assistant: ..." (assistant)
    - "User: ..." (user)
    """
    messages = []

    # Split on message markers, keeping the marker
    pattern = r'(?:^|\n)(A:|AI:|Assistant:|User:)\s*'
    parts = re.split(pattern, text.strip())

    # parts will be: [preamble, marker1, content1, marker2, content2, ...]
    # Skip any preamble before first marker
    i = 0
    while i < len(parts) and parts[i] not in ALL_MARKERS:
        i += 1

    while i < len(parts) - 1:
        marker = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if marker in ASSISTANT_MARKERS:
            role = 'assistant'
        else:
            role = 'user'

        if content:
            messages.append(Message(role=role, content=content))

        i += 2

    return messages
