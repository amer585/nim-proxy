# NVIDIA NIM Proxy — Cloudflare Worker

OpenAI-compatible proxy → NVIDIA NIM (`https://integrate.api.nvidia.com/v1`) for **GLM-5.2 (max thinking)** + DiffusionGemma.

## Deploy

```bash
npm install
npx wrangler login          # one-time
npx wrangler deploy
```

## Set secrets

```bash
npx wrangler secret put PROXY_AUTH_TOKEN
npx wrangler secret put NVIDIA_KEY_1
npx wrangler secret put NVIDIA_KEY_2
npx wrangler secret put NVIDIA_KEY_3
npx wrangler secret put NVIDIA_KEY_4
```
Or via the dashboard: **Workers & Pages → nim-proxy → Settings → Variables and Secrets → Add**.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible chat (streaming SSE) |
| GET | `/health` | `{"status":"ok","keys":4}` |
| GET | `/models` | JSON model list |

## Use with any OpenAI client (ZCode/OpenCode)

- **Base URL:** `https://nim-proxy.<your-subdomain>.workers.dev/v1`
- **API key:** your `PROXY_AUTH_TOKEN`
- **Model:** `z-ai/glm-5.2`

## What the proxy does

- GLM-5.2: `enable_thinking=true` + `reasoning_effort="max"` (deepest thinking, NO cap)
- Whitelist param filtering (strips `reasoning_effort`, `parallel_tool_calls`, `store`, etc. — NVIDIA rejects them)
- `max_tokens` floored to 202000 (prevents null-content when thinking eats all tokens)
- Multi-key rotation (NVIDIA_KEY_1–8), 15-min cooldown on 429, retry-with-backoff for DEGRADED
- Streaming: **immediate keepalive byte** + ping every 2s (prevents client timeouts during GLM's long thinking)

See `HANDOFF.md` for full architecture + troubleshooting.
