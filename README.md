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

## Batch Experiments (OpenAI v1)

Create a prompt and experiment config:

```bash
# Example files already included:
# - prompts/opacity_typology_coding.txt
# - experiments/exp0.0.yaml
```

Submit a batch experiment:

```bash
uv run python scripts/submit_experiment.py experiments/exp0.0.yaml
```

Submit a 25-transcript test run:

```bash
uv run python scripts/submit_experiment.py --test experiments/exp0.0.yaml
```

Collect status/results:

```bash
uv run python scripts/collect_batch_results.py experiments/exp0.0.yaml
```

Collect the test run:

```bash
uv run python scripts/collect_batch_results.py --test experiments/exp0.0.yaml
```

`collect_batch_results.py` now downloads raw output files and automatically writes `results.csv` when `output.jsonl` is available.

Run artifacts are written under `outputs/runs/<experiment_name>/`.
Test runs are written under `outputs/runs/<experiment_name>__test/`.

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set:
   - Main file path: `dashboard/app.py`
4. Deploy
