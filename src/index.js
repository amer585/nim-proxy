/**
 * NVIDIA NIM Proxy — Cloudflare Workers edition
 * ============================================
 * OpenAI-compatible proxy → NVIDIA NIM, for GLM-5.2 (max thinking) + DiffusionGemma.
 *
 * Preserved from the Python/FastAPI version:
 *  - GLM/Gemma: enable_thinking=true + reasoning_effort="max" (NO thinking cap)
 *  - Whitelist param filtering (strips reasoning_effort, parallel_tool_calls, store, etc.)
 *  - max_completion_tokens → max_tokens conversion
 *  - 202k token floor for thinking models (prevents null content)
 *  - Multi-key rotation + 15-min cooldown + retry-with-backoff
 *  - Bearer auth (PROXY_AUTH_TOKEN)
 *  - Streaming with IMMEDIATE keepalive byte + ping every 2s (prevents client timeout)
 */

const NVIDIA_URL = 'https://integrate.api.nvidia.com/v1';
const COOLDOWN = 15 * 60 * 1000; // 15 min (ms)
const RETRIES = 3;
const BACKOFF = 1000; // ms
const THINK_MIN_TOKENS = 202000;

// Only these params get forwarded to NVIDIA. Everything else the client sends
// (reasoning_effort, parallel_tool_calls, store, metadata, service_tier, etc.)
// is stripped — NVIDIA NIM rejects them.
const ALLOWED = new Set([
  'model', 'messages', 'tools', 'tool_choice',
  'temperature', 'top_p', 'top_k', 'max_tokens',
  'stream', 'seed', 'stop', 'response_format',
  'frequency_penalty', 'presence_penalty',
  'logprobs', 'top_logprobs', 'n',
  'chat_template_kwargs',
]);

// Module-level state (best-effort across Worker isolates). Key cooldown + RR.
let cooldown = {};
let rr = 0;

// ── Helpers ────────────────────────────────────────────────────────────────
function getKeys(env) {
  const keys = [];
  for (let i = 1; i <= 8; i++) {
    const k = env['NVIDIA_KEY_' + i];
    if (k && k.trim()) keys.push({ i, key: k.trim() });
  }
  return keys;
}

function jsonErr(status, message) {
  return new Response(JSON.stringify({ error: { message, type: 'error' } }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function checkAuth(authHeader, token) {
  if (!token) return jsonErr(401, 'PROXY_AUTH_TOKEN not configured.');
  if (!authHeader) return jsonErr(401, 'Missing Authorization header.');
  const parts = authHeader.split(' ');
  if (parts.length !== 2 || parts[0].toLowerCase() !== 'bearer')
    return jsonErr(401, 'Malformed Authorization header.');
  if (parts[1].trim() !== token) return jsonErr(401, 'Invalid token.');
  return null;
}

function buildPayload(body) {
  // 1. Flatten extra_body → root (NVIDIA rejects the wrapper)
  const extra = body.extra_body;
  if (extra && typeof extra === 'object' && !Array.isArray(extra)) {
    delete body.extra_body;
    Object.assign(body, extra);
  }

  // 2. Convert max_completion_tokens → max_tokens
  if ('max_completion_tokens' in body && !('max_tokens' in body)) {
    body.max_tokens = body.max_completion_tokens;
  }

  // 3. WHITELIST filter — only forward known-good params
  const filtered = {};
  for (const k of Object.keys(body)) {
    if (ALLOWED.has(k)) filtered[k] = body[k];
  }
  body = filtered;

  // 4. Strip top-level (OpenAI) reasoning_effort — NIM rejects it.
  //    NOTE: chat_template_kwargs.reasoning_effort is DIFFERENT (GLM-specific)
  //    and is set below. Only strip the top-level one here.
  delete body.reasoning_effort;

  // 5. GLM/Gemma thinking: enable_thinking ON, reasoning_effort="max", NO cap.
  //    User wants deepest thinking; time doesn't matter.
  const model = (body.model || '').toLowerCase();
  if (model.includes('glm') || model.includes('gemma')) {
    let ctk = body.chat_template_kwargs;
    if (ctk && typeof ctk === 'object' && !Array.isArray(ctk)) {
      if (ctk.enable_thinking === undefined) ctk.enable_thinking = true;
      if (!ctk.reasoning_effort) ctk.reasoning_effort = 'max';
      delete ctk.max_thinking_tokens; // no thinking bottleneck
      body.chat_template_kwargs = ctk;
    } else {
      body.chat_template_kwargs = { enable_thinking: true, reasoning_effort: 'max' };
    }

    // Enforce min max_tokens (parse int or string robustly).
    let mt = parseInt(body.max_tokens, 10);
    if (isNaN(mt)) mt = 0;
    if (mt < THINK_MIN_TOKENS) body.max_tokens = THINK_MIN_TOKENS;
  }

  return body;
}

/**
 * Send request with key rotation + retry-with-backoff for transient errors.
 * Returns { response } on success or { error: Response }.
 */
async function postWithRotation(url, payload, accept, env) {
  const keys = getKeys(env);
  if (keys.length === 0) return { error: jsonErr(503, 'No NVIDIA keys configured.') };

  let lastStatus = 0, lastDetail = 'No response.';

  for (let attempt = 0; attempt <= RETRIES; attempt++) {
    const n = keys.length;
    const order = [];
    for (let i = 0; i < n; i++) order.push(keys[(rr + i) % n]);
    rr = (rr + 1) % n;

    const now = Date.now();
    const avail = order.filter((e) => now >= (cooldown[e.i] || 0));
    if (avail.length === 0) {
      return { error: jsonErr(429, 'All keys cooling down (' + (COOLDOWN / 60000) + ' min). Retry later.') };
    }

    let transient = false;

    for (const e of avail) {
      const headers = {
        Authorization: 'Bearer ' + e.key,
        'Content-Type': 'application/json',
        Accept: accept,
        'Accept-Encoding': 'identity', // no compression — stream token-by-token
      };
      let resp;
      try {
        resp = await fetch(url, {
          method: 'POST',
          headers,
          body: JSON.stringify(payload),
        });
      } catch (ex) {
        lastStatus = 502;
        lastDetail = String(ex);
        transient = true;
        continue;
      }

      const status = resp.status;
      if (status === 429 || status >= 500) {
        if (status === 429) cooldown[e.i] = Date.now() + COOLDOWN;
        let detail;
        try {
          detail = (await resp.text()).slice(0, 2000);
        } catch {
          detail = 'HTTP ' + status;
        }
        lastDetail = detail;
        lastStatus = status;
        transient = true;
        continue;
      } else if (status >= 400) {
        let detail;
        try {
          detail = (await resp.text()).slice(0, 2000);
        } catch {
          detail = 'HTTP ' + status;
        }
        if (detail.toUpperCase().includes('DEGRADED')) {
          // Transient NVIDIA overload — retry next key.
          lastStatus = status;
          lastDetail = detail;
          transient = true;
          continue;
        }
        // Real client error — surface immediately.
        return { error: jsonErr(status, detail) };
      } else {
        return { response: resp };
      }
    }

    if (transient && attempt < RETRIES) {
      await new Promise((r) => setTimeout(r, BACKOFF * (attempt + 1)));
    }
  }

  return {
    error: jsonErr(
      lastStatus || 502,
      'All keys failed after ' + (RETRIES + 1) + ' tries. Last: HTTP ' + lastStatus + ' - ' + lastDetail
    ),
  };
}

// ── Worker entry ───────────────────────────────────────────────────────────
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // GET /health
    if (path === '/health' && request.method === 'GET') {
      return Response.json({ status: 'ok', keys: getKeys(env).length });
    }

    // GET /models
    if (path === '/models' && request.method === 'GET') {
      return Response.json({
        models: [
          { id: 'z-ai/glm-5.2', name: 'GLM-5.2', context: 1000000, note: 'Flagship. Max thinking. 1M context.' },
          { id: 'google/diffusiongemma-26b-a4b-it', name: 'DiffusionGemma-26B', context: 256000, note: 'Fast MoE. Thinking ON.' },
        ],
      });
    }

    // GET /  (minimal dashboard / status text)
    if (path === '/' && request.method === 'GET') {
      return new Response(
        'NVIDIA NIM Proxy — Cloudflare Worker\n\n' +
          'Endpoints:\n  POST /v1/chat/completions\n  GET  /health\n  GET  /models\n\n' +
          'Keys configured: ' + getKeys(env).length,
        { headers: { 'Content-Type': 'text/plain; charset=utf-8' } }
      );
    }

    // POST /v1/chat/completions
    if (path === '/v1/chat/completions' && request.method === 'POST') {
      const authErr = checkAuth(request.headers.get('authorization'), env.PROXY_AUTH_TOKEN);
      if (authErr) return authErr;

      let body;
      try {
        body = await request.json();
      } catch {
        return jsonErr(400, 'Invalid JSON.');
      }
      if (typeof body !== 'object' || Array.isArray(body))
        return jsonErr(400, 'Body must be a JSON object.');

      const payload = buildPayload(body);
      const isStream = payload.stream === true;
      const accept = isStream ? 'text/event-stream' : 'application/json';

      // ── Streaming: immediate keepalive byte BEFORE the NVIDIA request ──
      // The client gets a byte instantly (so it never "didn't even connect"),
      // then we start the request and pump chunks, with ping every 2s during
      // GLM's long thinking phase.
      if (isStream) {
        const enc = new TextEncoder();
        const stream = new ReadableStream({
          async start(controller) {
            // Immediate first byte
            controller.enqueue(enc.encode(': proxy-connected\n\n'));

            let stopped = false;
            const ping = setInterval(() => {
              if (!stopped) {
                try {
                  controller.enqueue(enc.encode(': ping\n\n'));
                } catch {
                  stopped = true;
                }
              }
            }, 2000);

            try {
              const result = await postWithRotation(
                NVIDIA_URL + '/chat/completions',
                payload,
                accept,
                env
              );
              if (result.error) {
                // Forward the error as an SSE data line so the client can parse it.
                const errBody = await result.error.text();
                try {
                  controller.enqueue(enc.encode('data: ' + errBody + '\n\n'));
                } catch {}
                return;
              }
              const reader = result.response.body.getReader();
              let firstByte = false;
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                if (!firstByte) {
                  firstByte = true;
                  stopped = true; // real data flowing — stop pinging
                }
                try {
                  controller.enqueue(value);
                } catch {
                  break; // client gone
                }
              }
            } catch {
              // swallow — stream just ends
            } finally {
              stopped = true;
              clearInterval(ping);
              try {
                controller.close();
              } catch {}
            }
          },
          cancel() {
            // client disconnected — nothing extra to clean (finally handles it)
          },
        });

        return new Response(stream, {
          headers: {
            'Content-Type': 'text/event-stream; charset=utf-8',
            'Cache-Control': 'no-cache, no-transform',
            Connection: 'keep-alive',
          },
        });
      }

      // ── Non-streaming ──
      const result = await postWithRotation(NVIDIA_URL + '/chat/completions', payload, accept, env);
      if (result.error) return result.error;
      const text = await result.response.text();
      const ct = result.response.headers.get('content-type') || 'application/json';
      return new Response(text, { headers: { 'Content-Type': ct } });
    }

    return jsonErr(404, 'Not found. Use /v1/chat/completions, /health, or /models.');
  },
};
