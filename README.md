# OpenAI Terminal Chat

Small terminal chat client built on the OpenAI Python SDK and the Responses API.

## Features

- Interactive chat mode with `/help`, `/reset`, and `/exit`
- One-shot mode for scripts and shell pipelines
- Environment-based configuration with CLI overrides
- Explicit configuration validation and readable error messages
- Configurable timeouts, retries, token limits, and streaming

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Usage

Interactive mode:

```bash
python main.py
```

One-shot mode:

```bash
python main.py --message "Explain recursion in one paragraph."
```

Streaming mode:

```bash
python main.py --stream
```
