# Interviewer

Mobile-friendly dashboard for exploring the [Anthropic Interviewer dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer) - 1,250 AI-conducted interviews with professionals about how they use AI in their work.

## Local Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set:
   - Main file path: `dashboard/app.py`
4. Deploy
