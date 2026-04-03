---
name: audio-dl-updater
description: Automate the collection, summarization, and posting of latest Audio Deep Learning research from arXiv (cs.SD, eess.AS) and Papers with Code.
---

# Audio DL Updater

A tool to keep the blog updated with the latest advancements in Audio Deep Learning.

## Workflow

1.  **Crawl**: Fetches latest papers from arXiv specific categories (`cs.SD`, `eess.AS`).
2.  **Summarize**: Uses Gemini (via REST API) to summarize abstract in Korean.
3.  **Post**: Generates a Jekyll-compatible markdown post in `_posts/`.

## Setup

-   Requires `requests` and `PyYAML`.
-   Requires `GEMINI_API_KEY` environment variable if automated summarization is desired.

## Usage

Run the script manually:
```bash
python scripts/update_audio_news.py
```

Or trigger the agent workflow:
`/audio_update`

## Automation

Configured to run via GitHub Actions `.github/workflows/audio_update.yml`.
