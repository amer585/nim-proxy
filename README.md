---
title: Nvidia NIM Proxy
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# NVIDIA NIM Proxy

OpenAI-compatible proxy in front of NVIDIA NIM (`https://integrate.api.nvidia.com/v1`)
with multi-key rotation and automatic GLM reasoning-parameter injection.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/health` | `{"status":"ok"}` |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (streaming SSE). Generic passthrough for any text model (GLM, etc.). |

## Required Space secrets

Set these in the Space **Settings → Variables and secrets** page (as *secrets*):

- `NVIDIA_KEY_1`, `NVIDIA_KEY_2`, `NVIDIA_KEY_3`, `NVIDIA_KEY_4`
- `PROXY_AUTH_TOKEN` (the bearer token your client must send)

## Behavior

- Forwards the exact OpenAI request body to NVIDIA NIM.
- Streams responses chunk-by-chunk (no buffering).
- Rotates across the 4 NVIDIA keys; on HTTP 429 it retries the next key and
  cool-downs the offending key for 15 minutes.
- For any model whose name contains `glm`, it injects
  `chat_template_kwargs: {"enable_thinking": true}` at the **root** of the body
  (NVIDIA NIM rejects the `extra_body` wrapper).
- Requires `Authorization: Bearer <PROXY_AUTH_TOKEN>`; otherwise returns 401.

## Run locally

```bash
pip install -r requirements.txt
NVIDIA_KEY_1=... PROXY_AUTH_TOKEN=... uvicorn main:app --host 0.0.0.0 --port 7860
```
