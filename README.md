# Mock AI API

A local mock API for testing applications that integrate with OpenAI and Anthropic-compatible endpoints. It returns deterministic responses and can simulate streaming, tool calls, API errors, embeddings, moderation, files, batches, and response retrieval without making requests to a real provider.

## Features

- OpenAI-compatible endpoints on port `8011`
- Anthropic-compatible endpoints on port `8012`
- Non-streaming and Server-Sent Events (SSE) streaming responses
- Scenario-based responses selected with a `:mock-*` model suffix
- Tool calls and tool-use streams
- Simulated rate limits and other API errors
- OpenAI models, completions, responses, embeddings, moderation, image, and audio endpoints
- Anthropic messages, token counting, batches, and file endpoints
- In-memory state with no API keys, accounts, or external services required

## Requirements

- Python 3.11+

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
mock-ai-api
```

For environments that prefer a requirements file, install the runtime dependencies with
`python3 -m pip install -r requirements.txt` and launch with `python3 server.py`.

The servers start at:

- OpenAI: <http://127.0.0.1:8011>
- Anthropic: <http://127.0.0.1:8012>

Use `--openai-port`, `--anthropic-port`, or `--reload` to customize the launch:

```bash
mock-ai-api --openai-port 9001 --anthropic-port 9002 --reload
# or: python3 server.py --openai-port 9001 --anthropic-port 9002 --reload
```

## Examples

Check both health endpoints:

```bash
curl http://127.0.0.1:8011/health
curl http://127.0.0.1:8012/health
```

Create an OpenAI-compatible chat completion:

```bash
curl http://127.0.0.1:8011/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gpt-4.1:mock-text",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Request a simulated tool call:

```bash
curl http://127.0.0.1:8011/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gpt-4.1:mock-tool-call",
    "messages": [{"role": "user", "content": "Write a file"}]
  }'
```

Create an Anthropic-compatible message:

```bash
curl http://127.0.0.1:8012/v1/messages \
  -H 'content-type: application/json' \
  -d '{
    "model": "claude-sonnet-4-20250514:mock-text",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Set `"stream": true` in supported requests to receive streaming events. Scenario names are defined in `app/scenarios.py` and exercised by the test suite; examples include `mock-text`, `mock-tool-stream`, `mock-delay`, `mock-429`, and `mock-stream-error`.

## Development

Run the test suite with:

```bash
python3 -m pytest -q
```

The test clients instantiate each application directly, so the servers do not need to be running when tests execute.
The SDK compatibility suite starts isolated local Uvicorn processes and is tested with
`openai==2.16.0` and `anthropic==0.76.0`.
Its process-level fixture relies on POSIX file-descriptor inheritance, so those integration
tests are skipped on non-POSIX platforms.

## Project layout

```text
app/
  openai_app.py       OpenAI-compatible routes
  anthropic_app.py    Anthropic-compatible routes
  openai/             OpenAI response builders and catalog
  anthropic/          Anthropic response builders and catalog
  scenarios.py        Mock scenario parsing
  schemas.py          Request models
server.py             Starts both mock servers
tests/                API behavior tests
```

## Limitations

This project is intended for local development and automated tests. It is not a production inference server: responses are mocked, state is process-local and in-memory, authentication is not implemented, and the supported API surface is intentionally incomplete.

## License

This project is licensed under the [MIT License](LICENSE).
