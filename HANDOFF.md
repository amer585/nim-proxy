# NVIDIA NIM Proxy — Complete Handoff

> **Everything you need to understand, use, and maintain this proxy.**

---

## 🎯 What This Is

A **FastAPI proxy** that sits between your coding client (ZCode, OpenCode, etc.) and **NVIDIA NIM** (NVIDIA's hosted inference API at `https://integrate.api.nvidia.com/v1`). The proxy:

- Accepts OpenAI-compatible requests (`/v1/chat/completions`)
- Forwards them to NVIDIA NIM
- Adds GLM-5.2-specific thinking parameters automatically
- Rotates across **4 NVIDIA API keys** with retry-on-rate-limit
- Streams responses back token-by-token with keepalive pings

**Public URL:** `https://amer224-api.hf.space`

---

## 🌐 Hugging Face Space

- **Repo:** https://huggingface.co/spaces/amer224/api
- **Branch:** `main`
- **SDK:** Docker (Python 3.11)
- **Port:** 7860
- **Latest commit:** check the repo for the current `sha` (run `git log` on clone)

### How to view / manage
- **Settings:** https://huggingface.co/spaces/amer224/api/settings
- **Files:** https://huggingface.co/spaces/amer224/api/tree/main
- **Build logs:** Settings → Logs
- **Factory reboot** (force rebuild): Settings → "Factory reboot" button

### Required Space Secrets (set at https://huggingface.co/spaces/amer224/api/settings)
| Secret | Required? | Notes |
|---|---|---|
| `NVIDIA_KEY_1` | ✅ Yes | First key, 40 RPM free tier |
| `NVIDIA_KEY_2` | ✅ Yes | Second key (rotated on 429) |
| `NVIDIA_KEY_3` | ✅ Yes | Third key |
| `NVIDIA_KEY_4` | ⚠️ Recommended | Fourth key (optional but better rotation) |
| `PROXY_AUTH_TOKEN` | ✅ Yes | Your bearer token; clients must send it in `Authorization: Bearer <token>` |
| `CONTEXT_LIMIT` | Optional | Override model context (e.g. `150000` to match ZCode's 150K window). `0` = use model window. |
| `KEY_COOLDOWN_SECONDS` | Optional | Default 900 (15 min). Time to wait after 429 before retrying a key. |
| `TRANSIENT_RETRIES` | Optional | Default 3. Retry rounds for transient errors (DEGRADED, 429, 5xx). |
| `BACKOFF_BASE` | Optional | Default 1.0s. Backoff multiplier between retry rounds. |
| `KEEPALIVE_INTERVAL` | Optional | Default 5.0s. How often to send SSE keepalive pings. |
| `THINK_MIN_TOKENS` | Optional | Default 202000. Min max_tokens for thinking models (GLM max). |

---

## 🛠️ Tech Stack

- **FastAPI** — async web framework
- **httpx** — async HTTP client with connection pooling
- **uvicorn[standard]** — ASGI server (with uvloop + httptools for speed)
- **Python 3.11-slim** Docker base

Total codebase: **~509 lines** in `main.py` (one file, no dependencies beyond FastAPI/httpx/uvicorn).

---

## 📁 Files (only 4 in the repo)

```
hf-space/
├── main.py              # All proxy logic (509 lines)
├── requirements.txt     # fastapi==0.115.0, uvicorn[standard]==0.30.6, httpx==0.27.2
├── Dockerfile           # Python 3.11-slim, port 7860, uvloop+httptools
├── README.md            # Has HF Space YAML frontmatter (sdk: docker, app_port: 7860)
└── HANDOFF.md           # This file
```

---

## 🧠 How It Works (Architecture)

### Request flow for `/v1/chat/completions`:
```
Client → POST /v1/chat/completions
   ↓
1. AUTH: Check Authorization: Bearer <PROXY_AUTH_TOKEN>
   ↓
2. PARSE: Read JSON body as raw dict (no Pydantic model)
   ↓
3. NORMALIZE (_build_payload):
   a. Flatten extra_body → root (NVIDIA rejects the wrapper)
   b. Convert max_completion_tokens → max_tokens
   c. WHITELIST filter: only forward known-good params
      (strips reasoning_effort, parallel_tool_calls, store, metadata, etc.)
   d. GLM/Gemma: inject enable_thinking=true, reasoning_effort=max
      (NO max_thinking_tokens cap — user wants deepest thinking)
   e. Enforce THINK_MIN_TOKENS floor (202k) to prevent null-content
   ↓
4. SEND (_post_with_rotation):
   - Round-robin across 4 keys
   - 401/429/5xx → rotate to next key
   - Real 429 → cool that key down for 15 min
   - All keys fail with DEGRADED/429/5xx → backoff 1s, 2s, 3s and retry
   - Returns the FIRST successful 2xx response (httpx.Response, stream=True)
   ↓
5. STREAM BACK (StreamingResponse):
   - IMMEDIATE first byte: ": proxy-connected\n\n" keepalive
   - Forwards every chunk from NVIDIA as-is
   - Proactive keepalive pings every 2s during long waits
   - Scans for exact input_tokens count to fix the context tracker
```

### Why we need a whitelist
OpenAI clients (ZCode, OpenCode, etc.) inject many parameters that NVIDIA NIM doesn't accept:
- `reasoning_effort` (top-level) → "Unsupported reasoning effort" 400 error
- `max_completion_tokens` (OpenAI uses this, NVIDIA wants `max_tokens`)
- `parallel_tool_calls`, `store`, `metadata`, `service_tier`, `prediction`, `user`, `n`

The whitelist filters ALL these out and only forwards known-good params. **This was a critical fix** — without it, ZCode was getting 400 errors on every request.

---

## 🔑 Key NVIDIA API Details

- **Base URL:** `https://integrate.api.nvidia.com/v1`
- **Auth:** `Authorization: Bearer <NVIDIA_KEY>`
- **GLM-5.2 model ID:** `z-ai/glm-5.2`
- **DiffusionGemma model ID:** `google/diffusiongemma-26b-a4b-it`
- **Rate limit:** 40 RPM per key (free tier)
- **Max output (GLM-5.2):** 131,072 tokens (but proxy floors at 202k to prevent null-content)
- **Context window (GLM-5.2):** 1,000,000 tokens
- **Context window (DiffusionGemma):** 256,000 tokens

### Get new API keys
1. Go to https://build.nvidia.com
2. Sign in (free NVIDIA Developer Program account)
3. Click your profile → "API Keys" → "Generate New Key"
4. Copy key (starts with `nvapi-...`)
5. Add as Space secret

---

## 🧪 How to Test Locally

### Prerequisites
- Python 3.11+
- An NVIDIA API key (free)

### Setup
```bash
cd hf-space
pip install -r requirements.txt

export NVIDIA_KEY_1="nvapi-YOUR_KEY_HERE"
export PROXY_AUTH_TOKEN="my-secret-token"
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"

uvicorn main:app --host 0.0.0.0 --port 7860
```

### Quick test
```bash
# Health check
curl http://localhost:7860/health
# → {"status":"ok","keys":1}

# List models
curl http://localhost:7860/models | python3 -m json.tool

# Chat with GLM
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-token" \
  -d '{
    "model": "z-ai/glm-5.2",
    "messages": [{"role": "user", "content": "Say hello in 2 words"}],
    "stream": false
  }'
```

---

## 🖥️ Using with ZCode / OpenCode / Any OpenAI Client

### Configuration
| Setting | Value |
|---|---|
| **Base URL** | `https://amer224-api.hf.space/v1` |
| **API Key** | Your `PROXY_AUTH_TOKEN` value |
| **Model** | `z-ai/glm-5.2` (or `google/diffusiongemma-26b-a4b-it` for speed) |

### ZCode "Add model" form
| Field | Value |
|---|---|
| Model ID | `z-ai/glm-5.2` |
| Context window | `150000` (matches ZCode's effective limit) |

### OpenCode equivalent
Set provider in `~/.config/opencode/config.json`:
```json
{
  "provider": {
    "nim-proxy": {
      "baseURL": "https://amer224-api.hf.space/v1",
      "apiKey": "<your PROXY_AUTH_TOKEN>",
      "models": {
        "z-ai/glm-5.2": {"context": 150000}
      }
    }
  }
}
```

---

## 📊 Dashboard (Web UI)

Open `https://amer224-api.hf.space` in a browser for a minimal dashboard showing:
- Health status (4 keys configured)
- API token usage (last request's input_tokens, with bar)
- Per-key status (active/cooling)
- Model cards (GLM-5.2, DiffusionGemma)

The dashboard polls `/status` every 5 seconds. No chat interface, no graph — just status.

---

## 🔧 Common Tasks

### Rotate the PROXY_AUTH_TOKEN
1. Set new value as Space secret: `PROXY_AUTH_TOKEN`
2. Update all clients to use the new token
3. (Optional) Factory reboot to apply

### Add a 5th NVIDIA key
1. Get a new key from build.nvidia.com
2. Add as Space secret: `NVIDIA_KEY_5`
3. The proxy auto-scans `NVIDIA_KEY_1` through `NVIDIA_KEY_8` on every request — just add the secret and restart

### Change the model list
Edit `MODELS` in `main.py` (around line 56):
```python
MODELS = [
    {"id": "z-ai/glm-5.2", "name": "GLM-5.2", "ctx": 1_000_000, "note": "..."},
    {"id": "google/diffusiongemma-26b-a4b-it", "name": "DiffusionGemma-26B", "ctx": 256_000, "note": "..."},
    # Add more here
]
```

### Adjust thinking behavior
Edit the GLM branch in `_build_payload` (around line 130):
```python
# Current: max effort, no cap
ctk.setdefault("enable_thinking", True)
ctk.setdefault("reasoning_effort", "max")
ctk.pop("max_thinking_tokens", None)

# For faster (balanced) thinking:
# ctk.setdefault("reasoning_effort", "high")
# ctk.setdefault("max_thinking_tokens", 1024)
```

### Update / debug
1. Edit `hf-space/main.py`
2. `git commit` + `git push` to trigger Space rebuild (2-3 min)
3. Check the Space logs in HF settings

---

## 🐛 Troubleshooting

### "DEGRADED function cannot be invoked" (HTTP 400)
This is **NVIDIA's infrastructure**, not the proxy. GLM-5.2's hosted function is temporarily overloaded. The proxy already retries with backoff across all 4 keys. Just retry later.

### ZCode shows "reconnecting 7/10"
The proxy now sends an immediate keepalive byte (0.35s) so this should be rare. If it still happens:
1. Check `/status` — are any keys in cooling state?
2. Try a different model (DiffusionGemma is faster)

### Null content / "no content" error
The proxy enforces `max_tokens: 202000` for thinking models to prevent this. If you still see it, NVIDIA might have a bug — retry.

### "Unsupported reasoning effort: max" 400 error
The whitelist should strip top-level `reasoning_effort`. If this appears, check that `body.pop("reasoning_effort", None)` is still in `_build_payload` (around line 125).

### Space is slow to rebuild / stuck
1. Try Settings → "Factory reboot" to force a clean rebuild
2. If still stuck, check HF status: https://status.huggingface.co
3. Nuclear option: delete the Space and recreate from the same git repo

### Local testing
```bash
# Run with debug logging
uvicorn main:app --host 0.0.0.0 --port 7860 --log-level debug
```

---

## 📜 Git History (recent commits)

Each commit fixes a specific issue. The current HEAD is the deployed version.

- `c29f706` — Immediate keepalive byte BEFORE starting NVIDIA request (fixes "didn't even connect")
- `da0b74e` — reasoning_effort=max, NO thinking cap (user choice)
- `694b7b6` — Proactive keepalive pings every 2s + max_thinking_tokens=1024 (later reverted)
- `9eb5584` — Whitelist + thinking + various speed optimizations
- `56888ec` — Force-push clean proxy (removed polluted Next.js files)
- `3b32b78` — String max_tokens type bug fix
- `00c01c6` — Removed chat/cosmos; restored lightweight token counter
- `5fab4fc` — Replaced Cosmos video route (didn't work)
- `5082e04` — Higher token floor (16384)

---

## 🔄 What NOT to Do

- **Don't** add Pydantic request models — the proxy accepts any field, breaking Pydantic would reject useful params
- **Don't** add an `/api/chat` route — the proxy is a passthrough, not a chatbot
- **Don't** add a video generation endpoint — NVIDIA doesn't host video gen on the public API
- **Don't** remove the `_STRIP_PARAMS` whitelist (it was renamed `_ALLOWED` but the principle is critical)
- **Don't** add `max_thinking_tokens` without asking the user — they want max thinking
- **Don't** poll the proxy with Hugging Face's preview app — that was a previous mistake that polluted the repo

---

## 🆘 If You're Stuck

1. **Check the dashboard:** https://amer224-api.hf.space (shows live key status)
2. **Check Space logs:** https://huggingface.co/spaces/amer224/api/settings → Logs
3. **Test the proxy directly:**
   ```bash
   curl https://amer224-api.hf.space/health
   curl -H "Authorization: Bearer YOUR_TOKEN" \
     -X POST https://amer224-api.hf.space/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"z-ai/glm-5.2","messages":[{"role":"user","content":"hi"}]}'
   ```
4. **Clone the repo and test locally** to see Python errors
5. **Force a rebuild** via Settings → Factory reboot

---

## 📞 Quick Reference Card

| Action | Command |
|---|---|
| Health check | `curl https://amer224-api.hf.space/health` |
| List models | `curl https://amer224-api.hf.space/models` |
| Live status | `curl https://amer224-api.hf.space/status` |
| GLM-5.2 chat (curl) | `curl -X POST https://amer224-api.hf.space/v1/chat/completions -H "Authorization: Bearer TOKEN" -H "Content-Type: application/json" -d '{"model":"z-ai/glm-5.2","messages":[{"role":"user","content":"hi"}]}'` |
| Dashboard | Open `https://amer224-api.hf.space` in browser |
| Settings | https://huggingface.co/spaces/amer224/api/settings |
| HF Space repo | https://huggingface.co/spaces/amer224/api |
| Logs | Settings → Logs tab |

---

**You have everything you need. The proxy is robust, fast, and working.**
