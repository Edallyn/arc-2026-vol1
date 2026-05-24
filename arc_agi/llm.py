import asyncio
import os
from typing import Any

import litellm
from asynciolimiter import Limiter
from dotenv import load_dotenv
from litellm import acompletion
from litellm import exceptions as litellm_exceptions

from arc_agi.types import Models

# Silence unnecessary litellm logs.
litellm.suppress_debug_info = True

# Ollama only processes one request at a time efficiently; this semaphore enforces that.
_ollama_semaphore = asyncio.Semaphore(1)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

RETRIES = 1
RETRY_DELAY_SEC = 15

limiters: dict[Models, Limiter] = {
    "groq/openai/gpt-oss-120b": Limiter(1.0),
    "openai/gpt-5": Limiter(1.0),
    "openai/gpt-5.1": Limiter(1.0),
    "xai/grok-4-fast": Limiter(1.0),
    "xai/grok-4": Limiter(1.0),
    "anthropic/claude-sonnet-4-5": Limiter(1.0),
    "anthropic/claude-haiku-4-5": Limiter(1.0),
    "gemini/gemini-2.5-pro": Limiter(2.0),
    "gemini/gemini-3-pro-preview": Limiter(1.0),
    "gemini/gemini-2.5-flash": Limiter(0.2),
    "gemini/gemini-3.1-flash-lite-preview": Limiter(0.25), # Saniyede ~4 istek (Dakikada 15 RPM'e denk gelir)
    "gemini/gemini-3-flash-preview": Limiter(0.08),
    "gemini/gemini-2.5-flash-lite": Limiter(0.16),
    "ollama/qwen3:4b": Limiter(1.0),
    "groq/qwen/qwen3-32b": Limiter(0.03),
    "ollama/qwen2.5-coder:7b": Limiter(1.0),
}

props: dict[Models, dict] = {
    "groq/openai/gpt-oss-120b": {},
    "openai/gpt-5": {"reasoning_effort": "high"},
    "openai/gpt-5.1": {"reasoning_effort": "high"},
    "xai/grok-4-fast": {},
    "xai/grok-4": {},
    "anthropic/claude-sonnet-4-5": {"thinking": {"type": "enabled", "budget_tokens": 32_000}},
    "anthropic/claude-haiku-4-5": {"thinking": {"type": "enabled", "budget_tokens": 32_000}},
    "gemini/gemini-2.5-pro": {"thinking": {"type": "enabled", "budget_tokens": 16_000}},
    "gemini/gemini-3-pro-preview": {},
    "gemini/gemini-2.5-flash": {},
    "gemini/gemini-3.1-flash-lite-preview": {},
    "gemini/gemini-3-flash-preview": {},
    "gemini/gemini-2.5-flash-lite": {},
    "ollama/qwen3:4b": {},
    "groq/qwen/qwen3-32b": {},
    "ollama/qwen2.5-coder:7b": {},
}


async def llm(
    model: Models,
    message: str,
    temperature,
    request_timeout: int | None,
    max_remaining_time: float | None,
    max_remaining_timeouts: int | None,
    problem_id: str | None = None,
    retries: int = RETRIES,
) -> tuple[str, float, float | None, int | None, int, int]:
    attempt = 1
    while attempt <= retries:
        await limiters[model].wait()

        current_request_timeout = request_timeout or 15 * 60
        if max_remaining_time is not None:
            current_request_timeout = min(current_request_timeout, max_remaining_time)

        is_ollama = model.startswith("ollama/")
        start_time = asyncio.get_event_loop().time()
        try:
            if is_ollama:
                api_key = "ollama"
                # Use OpenAI-compatible endpoint to avoid litellm's aiohttp Ollama path
                extra_kwargs = {"api_base": "http://localhost:11434/v1"}
            elif model.startswith("gemini/"):
                api_key = GEMINI_API_KEY
                extra_kwargs = {}
            elif model.startswith("groq/"):
                api_key = os.getenv("GROQ_API_KEY")
                extra_kwargs = {}
            else:
                api_key = None
                extra_kwargs = {}

            # For Ollama: remap "ollama/..." → "openai/..." to force the OpenAI-compatible endpoint
            litellm_model = ("openai/" + model[len("ollama/"):]) if is_ollama else model

            messages_to_send = [{"role": "user", "content": message}]

            if is_ollama:
                print(f"\n{'#'*60}")
                print(f"[LLM SEND] model={litellm_model} problem={problem_id} attempt={attempt}")
                print(f"[LLM SEND] messages count: {len(messages_to_send)}")
                for i, msg in enumerate(messages_to_send):
                    print(f"[LLM SEND] messages[{i}] role={msg['role']} content ({len(msg['content'])} chars):")
                    print(msg['content'][:1000])
                    if len(msg['content']) > 1000:
                        print(f"... (truncated)")
                print(f"[LLM SEND] extra_kwargs={extra_kwargs} props={props.get(model, {})}")
                print(f"{'#'*60}\n")

            async with (_ollama_semaphore if is_ollama else asyncio.Semaphore(1)):
                resp: Any = await acompletion(
                    model=litellm_model,
                    messages=messages_to_send,
                    temperature=temperature,
                    timeout=current_request_timeout,
                    num_retries=0,
                    api_key=api_key,
                    **extra_kwargs,
                    **props[model],
                )

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            if max_remaining_time is not None:
                max_remaining_time -= duration

            usage = None
            if hasattr(resp, "model_extra") and resp.model_extra:
                usage = resp.model_extra.get("usage")
            if usage is None and isinstance(resp, dict):
                usage = resp.get("usage")

            prompt_tokens = 0
            completion_tokens = 0

            if usage is not None:
                if hasattr(usage, "prompt_tokens"):
                    prompt_tokens = usage.prompt_tokens or 0
                elif isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0) or 0

                if hasattr(usage, "completion_tokens"):
                    completion_tokens = usage.completion_tokens or 0
                elif isinstance(usage, dict):
                    completion_tokens = usage.get("completion_tokens", 0) or 0

            raw_content = resp["choices"][0]["message"]["content"].strip()
            if is_ollama:
                print(f"\n{'#'*60}")
                print(f"[LLM RECV] problem={problem_id} duration={duration:.1f}s tokens={prompt_tokens}+{completion_tokens}")
                print(f"[LLM RECV] raw response ({len(raw_content)} chars):")
                print(raw_content[:2000])
                if len(raw_content) > 2000:
                    print(f"... (truncated)")
                print(f"{'#'*60}\n")
            return (
                raw_content,
                duration,
                max_remaining_time,
                max_remaining_timeouts,
                prompt_tokens,
                completion_tokens,
            )

        except (
            litellm_exceptions.RateLimitError,
            litellm_exceptions.InternalServerError,
            litellm_exceptions.ServiceUnavailableError,
            litellm_exceptions.APIConnectionError,
            litellm_exceptions.APIError,
            litellm.RouterRateLimitError,
            litellm.RouterRateLimitErrorBasic,
        ) as e:
            print(f"{problem_id or ''} Ignoring {type(e).__name__} and retrying attempt {attempt}: {e}")
            if attempt == retries:
                raise e
            attempt += 1
            await asyncio.sleep(RETRY_DELAY_SEC)
            continue

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            if max_remaining_time is not None:
                max_remaining_time -= duration

            if "Timeout" in str(e):
                if max_remaining_timeouts is not None:
                    max_remaining_timeouts -= 1
                    print(f"{problem_id or ''} Timed out. Remaining timeouts: {max_remaining_timeouts}")

                if max_remaining_timeouts is not None and max_remaining_timeouts <= 0:
                    raise RuntimeError("Exceeded timeouts allotted to the request")

                if attempt == retries:
                    return (
                        "Timeout",
                        duration,
                        max_remaining_time,
                        max_remaining_timeouts,
                        0,
                        0,
                    )

            if max_remaining_time is not None and max_remaining_time <= 0:
                raise RuntimeError("Exceeded time allotted to the request")

            if attempt == retries:
                print(f"{problem_id or ''} Max retry limit reached. Last exception during call:")
                print(str(e))
                raise e

            print(str(e))
            print(f"Exception during request for problem: {problem_id or ''}. Retry number {attempt}.")
            await asyncio.sleep(RETRY_DELAY_SEC)
            attempt += 1

    raise RuntimeError("Retries exceeded")
