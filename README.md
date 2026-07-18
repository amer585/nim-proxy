# NVIDIA NIM Proxy (Docker)

OpenAI-compatible proxy → NVIDIA NIM for **GLM-5.2 (max thinking)** + DiffusionGemma.

## Deploy (free: Koyeb / Render / Fly.io / any Docker host)

1. Fork/clone this repo
2. Connect it to your Docker host (e.g. Koyeb → "Create service from GitHub")
3. Set environment variables (secrets):
   - `NVIDIA_KEY_1`, `NVIDIA_KEY_2`, `NVIDIA_KEY_3`, `NVIDIA_KEY_4`
   - `PROXY_AUTH_TOKEN`
4. Expose port **7860**
5. Done. Use the assigned URL as your OpenAI base URL.

## Use with ZCode / OpenCode
- Base URL: `https://<your-app>.koyeb.app/v1`
- API Key: your `PROXY_AUTH_TOKEN`
- Model: `z-ai/glm-5.2`

See `HANDOFF.md` for full docs.
