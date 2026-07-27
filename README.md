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

Use `--openai-port` or `--anthropic-port` to customize the launch:

```bash
mock-ai-api --openai-port 9001 --anthropic-port 9002
# or: python3 server.py --openai-port 9001 --anthropic-port 9002
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

Set `"stream": true` in supported requests to receive streaming events.

## Scenarios

Append a scenario name to a model with `:mock-*`. The registry in
`app/scenarios.py` is the source of truth for scenario behavior, documented
examples, and advertised scenario models.

| Scenario | Provider | Endpoint | Response mode | Status | Example model | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `mock-text` | openai | `/v1/chat/completions` | text | — | `gpt-4.1:mock-text` | Returns a fixed greeting. |
| `mock-tool-call` | openai | `/v1/chat/completions` | tool call | — | `gpt-4.1:mock-tool-call` | Returns one function tool call. |
| `mock-tool-stream` | openai | `/v1/chat/completions` | stream | — | `gpt-4.1:mock-tool-stream` | Streams function tool-call deltas. |
| `mock-bad-tool-json` | openai | `/v1/chat/completions` | tool call | — | `gpt-4.1:mock-bad-tool-json` | Returns intentionally malformed tool arguments. |
| `mock-delay` | openai | `/v1/chat/completions` | delay | — | `gpt-4.1:mock-delay` | Adds deterministic streaming delay. |
| `mock-400` | openai | `/v1/chat/completions` | error | 400 | `gpt-4.1:mock-400` | Returns an invalid-request error. |
| `mock-401` | openai | `/v1/chat/completions` | error | 401 | `gpt-4.1:mock-401` | Returns an authentication error. |
| `mock-429` | openai | `/v1/chat/completions` | error | 429 | `gpt-4.1:mock-429` | Returns a rate-limit error. |
| `mock-500` | openai | `/v1/chat/completions` | error | 500 | `gpt-4.1:mock-500` | Returns an API error. |
| `mock-503` | openai | `/v1/chat/completions` | error | 503 | `gpt-4.1:mock-503` | Returns a service-unavailable error. |
| `mock-text` | anthropic | `/v1/messages` | text | — | `claude-sonnet-4-20250514:mock-text` | Returns a fixed greeting. |
| `mock-anthropic-stream` | anthropic | `/v1/messages` | stream | — | `claude-sonnet-4-20250514:mock-anthropic-stream` | Streams text in Anthropic event order. |
| `mock-tool-use` | anthropic | `/v1/messages` | tool call | — | `claude-sonnet-4-20250514:mock-tool-use` | Returns one tool-use content block. |
| `mock-tool-stream` | anthropic | `/v1/messages` | stream | — | `claude-sonnet-4-20250514:mock-tool-stream` | Streams tool-use JSON deltas. |
| `mock-delay` | anthropic | `/v1/messages` | delay | — | `claude-sonnet-4-20250514:mock-delay` | Adds deterministic streaming delay. |
| `mock-stream-error` | anthropic | `/v1/messages` | error | — | `claude-sonnet-4-20250514:mock-stream-error` | Emits an API error event after streaming begins. |
| `mock-400` | anthropic | `/v1/messages` | error | 400 | `claude-sonnet-4-20250514:mock-400` | Returns an invalid-request error. |
| `mock-401` | anthropic | `/v1/messages` | error | 401 | `claude-sonnet-4-20250514:mock-401` | Returns an authentication error. |
| `mock-429` | anthropic | `/v1/messages` | error | 429 | `claude-sonnet-4-20250514:mock-429` | Returns a rate-limit error. |
| `mock-500` | anthropic | `/v1/messages` | error | 500 | `claude-sonnet-4-20250514:mock-500` | Returns an API error. |
| `mock-503` | anthropic | `/v1/messages` | error | 503 | `claude-sonnet-4-20250514:mock-503` | Returns a service-unavailable error. |

### Extending scenarios

1. Add the scenario metadata to `SCENARIOS` in `app/scenarios.py`, including its provider, endpoint, response mode, optional error status, and example model.
2. Implement the provider-specific behavior in the relevant builder or streaming path.
3. The provider model catalog is generated from the registry; do not add a duplicate catalog entry by hand.
4. Add direct API tests, SDK coverage where the endpoint is SDK-supported, and update this table to describe the behavior.

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
  scenarios.py        Scenario registry and model-suffix parsing
  schemas.py          Request models
server.py             Starts both mock servers
tests/                API behavior tests
```

## Limitations

This project is intended for local development and automated tests. It is not a production inference server: responses are mocked, state is process-local and in-memory, authentication is not implemented, and the supported API surface is intentionally incomplete.

## License

This project is licensed under the [MIT License](LICENSE).
