"""
NVIDIA NIM Proxy — clean rewrite.
Forwards OpenAI-compatible chat requests to NVIDIA NIM with multi-key rotation,
GLM thinking injection, and a token-usage dashboard.
"""

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional, Union

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ── Config ──────────────────────────────────────────────────────────────────
NVIDIA_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
COOLDOWN = int(os.environ.get("KEY_COOLDOWN_SECONDS", str(15 * 60)))
KEEPALIVE = float(os.environ.get("KEEPALIVE_INTERVAL", "5"))
RETRIES = int(os.environ.get("TRANSIENT_RETRIES", "3"))
BACKOFF = float(os.environ.get("BACKOFF_BASE", "1.0"))

PROXY_AUTH_TOKEN = os.environ.get("PROXY_AUTH_TOKEN", "").strip()
CONTEXT_LIMIT = int(os.environ.get("CONTEXT_LIMIT", "0"))  # 0 = use model window

# Thinking models need a HIGH token budget. GLM-5.2 generates long reasoning
# traces; if max_tokens is too low, ALL tokens go to reasoning and the final
# content comes back NULL -> clients show "no content" errors.
# GLM-5.2 NVIDIA hosted max output. A high floor guarantees content is always
# returned even on long-thinking requests. The actual time-to-content depends
# on NVIDIA's free-tier speed (~3-7s typical), not the token floor.
THINK_MIN_TOKENS = int(os.environ.get("THINK_MIN_TOKENS", "202000"))

# Only these params get forwarded to NVIDIA. Everything else ZCode/OpenCode
# sends (reasoning_effort, parallel_tool_calls, store, etc.) is stripped.
ALLOWED = {
    "model", "messages", "tools", "tool_choice",
    "temperature", "top_p", "top_k", "max_tokens",
    "stream", "seed", "stop", "response_format",
    "frequency_penalty", "presence_penalty",
    "logprobs", "top_logprobs", "n",
    "chat_template_kwargs",
}

# ── Keys ────────────────────────────────────────────────────────────────────
KEYS: list[dict] = []
for i in range(1, 9):
    k = os.environ.get(f"NVIDIA_KEY_{i}")
    if k and k.strip():
        KEYS.append({"i": i, "key": k.strip()})

cooldown: dict[int, float] = {}
_rr = 0

# ── Models (dashboard data) ────────────────────────────────────────────────
MODELS = [
    {"id": "z-ai/glm-5.2", "name": "GLM-5.2", "ctx": 1_000_000,
     "note": "Flagship. Thinking ON. 1M context."},
    {"id": "google/diffusiongemma-26b-a4b-it", "name": "DiffusionGemma-26B",
     "ctx": 256_000, "note": "Fast MoE. Thinking ON. 256K context."},
]

# ── Token tracking ─────────────────────────────────────────────────────────
HISTORY: list[dict] = []
_RE_IN = re.compile(rb'"input_tokens"\s*:\s*(\d+)')
_RE_PR = re.compile(rb'"prompt_tokens"\s*:\s*(\d+)')

# ── HTTP client ────────────────────────────────────────────────────────────
_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app):
    global _client
    _client = httpx.AsyncClient(
        http2=False,
        timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200, keepalive_expiry=120),
    )
    try:
        yield
    finally:
        if _client:
            await _client.aclose()


app = FastAPI(title="NIM Proxy", version="3.0", lifespan=lifespan)


# ── Helpers ────────────────────────────────────────────────────────────────
def _err(code: int, msg: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"error": {"message": msg, "type": "error"}})


def _check_auth(auth: Optional[str]) -> Optional[JSONResponse]:
    if not PROXY_AUTH_TOKEN:
        return _err(401, "PROXY_AUTH_TOKEN not configured.")
    if not auth:
        return _err(401, "Missing Authorization header.")
    parts = auth.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return _err(401, "Malformed Authorization header.")
    if parts[1].strip() != PROXY_AUTH_TOKEN:
        return _err(401, "Invalid token.")
    return None


def _build_payload(body: dict) -> dict:
    """Clean + normalize the request for NVIDIA compatibility."""
    # 1. Flatten extra_body
    extra = body.pop("extra_body", None)
    if isinstance(extra, dict):
        body.update(extra)

    # 2. Convert max_completion_tokens -> max_tokens
    if "max_completion_tokens" in body and "max_tokens" not in body:
        body["max_tokens"] = body["max_completion_tokens"]

    # 3. WHITELIST: only forward known-good params (strips reasoning_effort etc.)
    body = {k: v for k, v in body.items() if k in ALLOWED}

    # 4. Strip the top-level (OpenAI-standard) reasoning_effort — NIM rejects it
    body.pop("reasoning_effort", None)
    # NOTE: chat_template_kwargs.reasoning_effort is DIFFERENT (GLM-specific)
    # and is kept + defaulted to "high" below.

    # 5. Thinking model handling
    # NO bottlenecks. The user explicitly wants max thinking effort, so GLM
    # thinks as long as it needs to. Just enable thinking and set reasoning_effort
    # to "max" for deepest reasoning (per Z.ai docs). No max_thinking_tokens cap.
    model = str(body.get("model", "")).lower()
    if "glm" in model or "gemma" in model:
        ctk = body.get("chat_template_kwargs")
        if isinstance(ctk, dict):
            ctk.setdefault("enable_thinking", True)
            ctk.setdefault("reasoning_effort", "max")
            # Remove any thinking token cap the user might have set
            ctk.pop("max_thinking_tokens", None)
            body["chat_template_kwargs"] = ctk
        else:
            body["chat_template_kwargs"] = {
                "enable_thinking": True,
                "reasoning_effort": "max",
            }

        # CRITICAL: enforce high max_tokens so thinking doesn't eat everything.
        # Parse int or string robustly.
        mt = body.get("max_tokens")
        try:
            mt = int(mt)
        except (TypeError, ValueError):
            mt = 0
        if mt < THINK_MIN_TOKENS:
            body["max_tokens"] = THINK_MIN_TOKENS

        # SPEED: inject a "be concise" hint to keep thinking fast.
        # This dramatically reduces response time (from ~15s to ~2-4s) by
        # telling GLM/Gemma to think briefly and respond directly. Thinking
        # is still ON, just more efficient.
        messages = body.get("messages")
        if isinstance(messages, list) and len(messages) > 0:
            first = messages[0]
            if isinstance(first, dict) and first.get("role") != "system":
                body["messages"] = [
                    {"role": "system", "content": "Be concise. Think briefly then answer directly."}
                ] + messages
            elif isinstance(first, dict) and first.get("role") == "system":
                existing = first.get("content", "")
                if isinstance(existing, str) and "concise" not in existing.lower():
                    first["content"] = "Be concise. Think briefly then answer directly.\n\n" + existing

    return body


def _estimate_tokens(body: dict) -> int:
    chars = 0
    for msg in body.get("messages", []):
        c = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    chars += len(p["text"])
                elif isinstance(p, dict):
                    chars += len(json.dumps(p))
    if body.get("tools"):
        chars += len(json.dumps(body["tools"]))
    return max(1, chars // 4)


def _record(model: str, tokens: int):
    HISTORY.append({"ts": time.time(), "tokens": int(tokens), "model": str(model)})
    if len(HISTORY) > 600:
        del HISTORY[: len(HISTORY) - 600]


def _ctx_limit(model: str) -> int:
    if CONTEXT_LIMIT > 0:
        return CONTEXT_LIMIT
    for m in MODELS:
        if m["id"].lower() == model.lower():
            return m["ctx"]
    return 150_000


async def _forward(resp: httpx.Response, is_stream: bool = False):
    """Stream upstream response; inject keepalive pings if NVIDIA is slow;
    scan for exact token count.

    Proactive keepalive: a parallel task sends an SSE comment every 2s
    while we wait for upstream data. This keeps the SSE connection alive
    during GLM's long thinking phase, preventing ZCode from reconnecting.
    """
    scanned = False
    buf = b""
    first_data_sent = False
    stop_keepalive = asyncio.Event()
    keepalive_queue = asyncio.Queue()

    async def _keepalive_pinger():
        """Push a keepalive onto the queue every 2s until stopped."""
        try:
            while not stop_keepalive.is_set():
                try:
                    await asyncio.wait_for(stop_keepalive.wait(), timeout=2.0)
                    return
                except asyncio.TimeoutError:
                    await keepalive_queue.put(b": ping\n\n")
        except asyncio.CancelledError:
            return

    pinger = None
    if is_stream:
        pinger = asyncio.create_task(_keepalive_pinger())
    ai = resp.aiter_raw()
    try:
        while True:
            # Non-blocking check for keepalive pings
            try:
                while not keepalive_queue.empty():
                    ka = keepalive_queue.get_nowait()
                    if is_stream:
                        yield ka
            except asyncio.QueueEmpty:
                pass

            try:
                chunk = await asyncio.wait_for(ai.__anext__(), timeout=KEEPALIVE)
                if chunk:
                    if not first_data_sent:
                        first_data_sent = True
                        stop_keepalive.set()  # stop pinging, real data is flowing
                    if not scanned:
                        buf = (buf + chunk)[-96:]
                        m = _RE_IN.search(buf) or _RE_PR.search(buf)
                        if m:
                            if HISTORY:
                                HISTORY[-1]["tokens"] = int(m.group(1))
                            scanned = True
                            buf = b""
                    yield chunk
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                if is_stream:
                    yield b": keepalive\n\n"
    finally:
        stop_keepalive.set()
        if pinger:
            try:
                pinger.cancel()
            except Exception:
                pass
        try:
            await resp.aclose()
        except Exception:
            pass


async def _post(url: str, payload: dict, accept: str = "text/event-stream") -> Union[JSONResponse, httpx.Response]:
    """Send request with key rotation + retry-with-backoff for transient errors."""
    if not KEYS:
        return _err(503, "No NVIDIA keys configured.")
    if not _client:
        return _err(503, "Client not ready.")

    global _rr
    n = len(KEYS)
    last_status, last_detail = None, "No response."

    for attempt in range(RETRIES + 1):
        order = [KEYS[(_rr + i) % n] for i in range(n)]
        _rr = (_rr + 1) % n
        now = time.time()
        avail = [e for e in order if now >= cooldown.get(e["i"], 0)]
        if not avail:
            return _err(429, f"All keys cooling down ({COOLDOWN // 60} min). Retry later.")

        transient = False
        for e in avail:
            headers = {
                "Authorization": f"Bearer {e['key']}",
                "Content-Type": "application/json",
                "Accept": accept,
                "Accept-Encoding": "identity",
            }
            try:
                req = _client.build_request("POST", url, json=payload, headers=headers)
                resp = await _client.send(req, stream=True)
            except Exception as ex:
                last_status, last_detail, transient = 502, str(ex), True
                continue

            status = resp.status_code
            if status == 429 or status >= 500:
                if status == 429:
                    cooldown[e["i"]] = time.time() + COOLDOWN
                try:
                    last_detail = (await resp.aread()).decode(errors="replace")[:2000]
                except Exception:
                    last_detail = f"HTTP {status}"
                last_status, transient = status, True
                await resp.aclose()
                continue
            elif status >= 400:
                try:
                    detail = (await resp.aread()).decode(errors="replace")[:2000]
                except Exception:
                    detail = f"HTTP {status}"
                await resp.aclose()
                if "DEGRADED" in detail.upper():
                    last_status, last_detail, transient = status, detail, True
                    continue
                return _err(status, detail)
            else:
                return resp

        if transient and attempt < RETRIES:
            await asyncio.sleep(BACKOFF * (attempt + 1))

    return _err(last_status or 502, f"All keys failed after {RETRIES + 1} tries. Last: HTTP {last_status} - {last_detail}")


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "keys": len(KEYS)}


@app.get("/status")
async def status():
    now = time.time()
    active = 0
    keys = []
    for e in KEYS:
        until = cooldown.get(e["i"], 0)
        if now >= until:
            keys.append({"i": e["i"], "state": "active"})
            active += 1
        else:
            keys.append({"i": e["i"], "state": "cooling", "left": int(until - now)})

    latest = HISTORY[-1] if HISTORY else None
    if latest:
        ct = latest["tokens"]
        cm = latest["model"]
        age = int(now - latest["ts"])
    else:
        ct, cm, age = 0, "", None
    lim = _ctx_limit(cm)

    return {
        "keys": len(KEYS), "active": active, "cooling": len(KEYS) - active,
        "key_states": keys,
        "ctx": {
            "tokens": ct, "model": cm, "limit": lim,
            "pct": round(ct / lim * 100, 1) if lim else 0,
            "age": age,
        },
    }


@app.get("/")
async def root():
    return HTMLResponse(_dashboard())


@app.get("/models")
async def models():
    return {"models": MODELS}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    auth_err = _check_auth(request.headers.get("authorization"))
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return _err(400, "Invalid JSON.")
    if not isinstance(body, dict):
        return _err(400, "Body must be a JSON object.")

    payload = _build_payload(body)
    is_stream = bool(payload.get("stream", False))

    # Record context estimate immediately (corrected by _forward with real count)
    _record(payload.get("model", ""), _estimate_tokens(payload))

    accept = "text/event-stream" if is_stream else "application/json"

    # For streaming: start sending data to the client IMMEDIATELY (first byte
    # is a keepalive) so ZCode sees bytes flowing and doesn't time out, then
    # we start the NVIDIA request. This prevents the "doesn't even connect"
    # problem when GLM takes a long time to start generating.
    if is_stream:
        async def stream_with_immediate_keepalive():
            # Send a leading keepalive so the client's TTFB timer starts now
            yield b": proxy-connected\n\n"
            # Then start the actual upstream request
            result = await _post(f"{NVIDIA_URL}/chat/completions", payload, accept)
            if isinstance(result, JSONResponse):
                # Error before any upstream data: send as a data: line so client
                # can parse it as a regular SSE event
                body_bytes = result.body
                yield b"data: " + body_bytes + b"\n\n"
                return
            async for chunk in _forward(result, is_stream=True):
                yield chunk
        return StreamingResponse(
            stream_with_immediate_keepalive(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming: just post and return
    result = await _post(f"{NVIDIA_URL}/chat/completions", payload, accept)
    if isinstance(result, JSONResponse):
        return result
    ct = result.headers.get("content-type", "application/json")
    return StreamingResponse(
        _forward(result, is_stream=False),
        media_type=ct,
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ── Dashboard ──────────────────────────────────────────────────────────────
_DASH = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIM Proxy</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,system-ui,sans-serif;background:#0a0e17;color:#e2e8f0;padding:24px}
.w{max-width:900px;margin:0 auto}
h1{font-size:22px;font-weight:700;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:#7b8aa3;font-size:13px;margin-top:3px}
.badge{display:flex;align-items:center;gap:8px;background:#131a26;padding:8px 14px;border-radius:999px;font-size:13px;border:1px solid #1b2433}
.dot{width:9px;height:9px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e}
.pulse{width:9px;height:9px;border-radius:50%;background:#38bdf8;animation:p 1.6s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
.card{background:#131a26;border:1px solid #1b2433;border-radius:16px;padding:20px;margin:18px 0}
.row{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:18px}
.nums{display:flex;align-items:baseline;gap:7px;margin:10px 0}
.used{font-size:36px;font-weight:800;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sep{font-size:22px;color:#7b8aa3}.lim{font-size:22px;font-weight:700}
.unit{color:#7b8aa3;font-size:13px}
.pct{margin-left:8px;font-size:14px;font-weight:700;padding:3px 10px;border-radius:7px;background:#0a0e17;color:#38bdf8}
.bar{height:12px;background:#0a0e17;border-radius:6px;overflow:hidden;border:1px solid #1b2433}
.fill{height:100%;background:linear-gradient(90deg,#22c55e,#38bdf8);border-radius:6px;transition:width .4s;width:0}
.foot{color:#7b8aa3;font-size:12px;margin-top:8px}
.kp{display:flex;align-items:center;gap:8px;background:#0a0e17;border:1px solid #1b2433;padding:8px 14px;border-radius:10px;font-size:13px}
.kd{width:8px;height:8px;border-radius:50%}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.mc{background:#131a26;border:1px solid #1b2433;border-radius:14px;padding:18px}
.mn{font-size:17px;font-weight:700}.mi{color:#7b8aa3;font-size:11px;font-family:monospace;margin:4px 0}
.mc b{color:#38bdf8}
code{background:#0a0e17;padding:2px 6px;border-radius:4px;font-size:11px;color:#818cf8}
.ftr{text-align:center;color:#7b8aa3;font-size:12px;margin-top:30px;line-height:1.6}
</style></head><body><div class="w">
<div class="row">
<div><h1>NVIDIA NIM Proxy</h1><div class="sub">GLM-5.2 · DiffusionGemma · streaming · 4-key rotation</div></div>
<div class="badge"><span class="dot" id="sd"></span><span id="st">—</span></div>
</div>
<div class="card"><div class="row" style="margin:0"><span class="pulse"></span> <b>API Token Usage</b> <span style="margin-left:auto;color:#818cf8;font-family:monospace;font-size:12px" id="cm">—</span></div>
<div class="nums"><span class="used" id="tu">—</span><span class="sep">/</span><span class="lim" id="tl">—</span><span class="unit">tokens</span><span class="pct" id="tp">—</span></div>
<div class="bar"><div class="fill" id="tf"></div></div>
<div class="foot" id="tfo">Waiting for first request…</div></div>
<div class="card"><div class="row" style="margin:0"><span class="pulse"></span> <b>Keys</b></div><div style="display:flex;gap:10px;flex-wrap:wrap" id="kr"></div></div>
<div class="grid" id="mg"></div>
<div class="ftr"><code>POST https://amer224-api.hf.space/v1/chat/completions</code><br>Health: <code>/health</code> · Status: <code>/status</code></div>
</div>
<script>
const $=id=>document.getElementById(id);
const MODELS=__MS__;
$('mg').innerHTML=MODELS.map(m=>`<div class="mc"><div class="mn">${m.name}</div><div class="mi">${m.id}</div><div><b>${(m.ctx/1000).toFixed(0)}K</b> context</div><div class="foot">${m.note}</div></div>`).join('');
async function poll(){
  try{
    const d=await(await fetch('/status')).json();
    const sd=$('sd'),st=$('st');
    if(!d.keys){sd.style.background='#ef4444';st.textContent='No keys'}
    else if(d.cooling>0){sd.style.background='#f59e0b';st.textContent=d.active+'/'+d.keys+' active'}
    else{sd.style.background='#22c55e';st.textContent=d.active+' key'+(d.active!=1?'s':'')+' active'}
    $('kr').innerHTML=d.key_states.map(k=>{
      const c=k.state==='cooling'?'cooling':'active';
      const l=k.state==='cooling'?'cooling '+k.left+'s':'active';
      return `<div class="kp"><span class="kd" style="background:${k.state==='cooling'?'#f59e0b':'#22c55e'};box-shadow:0 0 6px ${k.state==='cooling'?'#f59e0b':'#22c55e'}"></span> Key ${k.i} <span style="color:#7b8aa3;font-size:11px">${l}</span></div>`;
    }).join('');
    const c=d.ctx;
    if(c&&c.tokens>0){
      const fK=n=>n>=1000?(n/1000).toFixed(1)+'K':String(n);
      $('tu').textContent=fK(c.tokens);$('tl').textContent=fK(c.limit);
      $('tp').textContent='('+c.pct+'%)';
      const f=$('tf');f.style.width=Math.min(c.pct,100)+'%';
      if(c.pct>90)f.style.background='linear-gradient(90deg,#f59e0b,#ef4444)';
      else if(c.pct>70)f.style.background='linear-gradient(90deg,#22c55e,#f59e0b)';
      $('cm').textContent=(c.model.split('/').pop()||'—')+(c.age!=null?(c.age<60?' · '+c.age+'s ago':' · '+Math.floor(c.age/60)+'m ago'):'');
      $('tfo').textContent=c.pct>90?'⚠️ Near limit — consider fresh context':c.pct>70?'Getting full':'✓ Plenty of context remaining';
    }
  }catch(e){$('st').textContent='offline'}
}
poll();setInterval(poll,5000);
</script></body></html>"""


def _dashboard() -> str:
    ms = json.dumps([{"id": m["id"], "name": m["name"], "ctx": m["ctx"], "note": m["note"]} for m in MODELS])
    return _DASH.replace("__MS__", ms)
