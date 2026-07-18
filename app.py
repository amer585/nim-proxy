"""
NVIDIA NIM Proxy — Gradio/FastAPI Edition
Runs FOR FREE on Hugging Face Spaces (CPU Basic, no credit card needed).
"""

import asyncio
import json
import os
import re
import time
from typing import Optional, Union

import httpx
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── Config ──────────────────────────────────────────────────────────────────
NVIDIA_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
COOLDOWN = int(os.environ.get("KEY_COOLDOWN_SECONDS", str(15 * 60)))
KEEPALIVE = float(os.environ.get("KEEPALIVE_INTERVAL", "5"))
RETRIES = int(os.environ.get("TRANSIENT_RETRIES", "3"))
BACKOFF = float(os.environ.get("BACKOFF_BASE", "1.0"))

PROXY_AUTH_TOKEN = os.environ.get("PROXY_AUTH_TOKEN", "").strip()
THINK_MIN_TOKENS = int(os.environ.get("THINK_MIN_TOKENS", "202000"))

ALLOWED = {
    "model", "messages", "tools", "tool_choice",
    "temperature", "top_p", "top_k", "max_tokens",
    "stream", "seed", "stop", "response_format",
    "frequency_penalty", "presence_penalty",
    "logprobs", "top_logprobs", "n",
    "chat_template_kwargs",
}

KEYS: list[dict] = []
for i in range(1, 9):
    k = os.environ.get(f"NVIDIA_KEY_{i}")
    if k and k.strip():
        KEYS.append({"i": i, "key": k.strip()})

cooldown: dict[int, float] = {}
_rr = 0

# ── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI()

def _err(code: int, msg: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"error": {"message": msg, "type": "error"}})

def _check_auth(auth: Optional[str]) -> Optional[JSONResponse]:
    if not PROXY_AUTH_TOKEN: return _err(401, "PROXY_AUTH_TOKEN not configured.")
    if not auth: return _err(401, "Missing Authorization header.")
    parts = auth.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer": return _err(401, "Malformed Authorization header.")
    if parts[1].strip() != PROXY_AUTH_TOKEN: return _err(401, "Invalid token.")
    return None

def _build_payload(body: dict) -> dict:
    extra = body.pop("extra_body", None)
    if isinstance(extra, dict): body.update(extra)

    if "max_completion_tokens" in body and "max_tokens" not in body:
        body["max_tokens"] = body["max_completion_tokens"]

    body = {k: v for k, v in body.items() if k in ALLOWED}
    body.pop("reasoning_effort", None)

    model = str(body.get("model", "")).lower()
    if "glm" in model or "gemma" in model:
        ctk = body.get("chat_template_kwargs")
        if isinstance(ctk, dict):
            ctk.setdefault("enable_thinking", True)
            ctk.setdefault("reasoning_effort", "max")
            ctk.pop("max_thinking_tokens", None)
            body["chat_template_kwargs"] = ctk
        else:
            body["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": "max"}

        mt = body.get("max_tokens")
        try: mt = int(mt)
        except: mt = 0
        if mt < THINK_MIN_TOKENS: body["max_tokens"] = THINK_MIN_TOKENS

    return body

async def _post(url, payload, accept="text/event-stream"):
    if not KEYS: return _err(503, "No NVIDIA keys configured.")
    global _rr
    last_status, last_detail = None, "No response."

    for attempt in range(RETRIES + 1):
        order = [KEYS[(_rr + i) % len(KEYS)] for i in range(len(KEYS))]
        _rr = (_rr + 1) % len(KEYS)
        now = time.time()
        avail = [e for e in order if now >= cooldown.get(e["i"], 0)]
        if not avail: return _err(429, f"All keys cooling down ({COOLDOWN // 60} min). Retry later.")

        transient = False
        for e in avail:
            headers = {"Authorization": f"Bearer {e['key']}", "Content-Type": "application/json", "Accept": accept, "Accept-Encoding": "identity"}
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except Exception as ex:
                last_status, last_detail, transient = 502, str(ex), True
                continue

            status = resp.status_code
            if status == 429 or status >= 500:
                if status == 429: cooldown[e["i"]] = time.time() + COOLDOWN
                last_detail, last_status, transient = resp.text[:2000], status, True
                continue
            elif status >= 400:
                detail = resp.text[:2000]
                if "DEGRADED" in detail.upper():
                    last_status, last_detail, transient = status, detail, True
                    continue
                return _err(status, detail)
            else:
                return resp

        if transient and attempt < RETRIES:
            await asyncio.sleep(BACKOFF * (attempt + 1))

    return _err(last_status or 502, f"All keys failed after {RETRIES + 1} tries. Last: HTTP {last_status} - {last_detail}")

async def _forward(resp: httpx.Response):
    async for chunk in resp.aiter_raw():
        yield chunk

@app.post("/v1/chat/completions")
async def chat(request: Request):
    auth_err = _check_auth(request.headers.get("authorization"))
    if auth_err: return auth_err
    try: body = await request.json()
    except: return _err(400, "Invalid JSON.")
    if not isinstance(body, dict): return _err(400, "Body must be a JSON object.")

    payload = _build_payload(body)
    is_stream = bool(payload.get("stream", False))

    if is_stream:
        async def stream_with_keepalive():
            yield b": proxy-connected\n\n"
            result = await _post(f"{NVIDIA_URL}/chat/completions", payload)
            if isinstance(result, JSONResponse):
                yield b"data: " + result.body + b"\n\n"
                return
            async for chunk in _forward(result):
                yield chunk
        return StreamingResponse(stream_with_keepalive(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    result = await _post(f"{NVIDIA_URL}/chat/completions", payload)
    if isinstance(result, JSONResponse): return result
    return Response(result.content, media_type=result.headers.get("content-type", "application/json"))

@app.get("/health")
async def health(): return {"status": "ok", "keys": len(KEYS)}

@app.get("/models")
async def models():
    return {"data": [{"id": "z-ai/glm-5.2", "object": "model"}, {"id": "google/diffusiongemma-26b-a4b-it", "object": "model"}]}

# ── Gradio UI Wrapper ──────────────────────────────────────────────────────
# We mount a dummy Gradio app so Hugging Face runs it on the Free CPU tier.
with gr.Blocks(title="NIM Proxy") as demo:
    gr.Markdown("# ✅ NVIDIA NIM Proxy is Running")
    gr.Markdown("API is accessible at `/v1/chat/completions`")

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
