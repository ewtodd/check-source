"""Minimal JSON-over-HTTP client for an OpenAI-compatible endpoint.

Only the standard library is used: the server tokenizes and generates, the
client builds prompts, posts them and reads the answers back.
"""

import json
import time
import urllib.request


class Endpoint:
    def __init__(self, base_url, timeout=3600, retries=5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def post(self, path, payload):
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        last_error = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode())
            except Exception as error:  # noqa: BLE001 - retried, then raised
                last_error = error
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError("request failed after %d tries: %s" % (self.retries, last_error))

    def chat(self, model, prompt, generation, seed=None, template_kwargs=None):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        payload.update(generation)
        if seed is not None:
            payload["seed"] = seed
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        return self.post("/v1/chat/completions", payload)

    def complete(self, model, prompt, generation, stop=None, seed=None):
        payload = {"model": model, "prompt": prompt, "n": 1}
        if stop:
            payload["stop"] = stop
        payload.update(generation)
        if seed is not None:
            payload["seed"] = seed
        return self.post("/v1/completions", payload)


def chat_reply(response):
    """Pull the answer, thinking trace, finish reason and token counts out of a
    chat-completions response.

    vLLM 0.29 exposes the parsed thinking trace as `reasoning`; other builds and
    servers use `reasoning_content`.
    """
    choice = response["choices"][0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "text": message.get("content") or "",
        "reasoning": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
    }


def completion_text(response):
    choice = response["choices"][0]
    usage = response.get("usage") or {}
    return {
        "text": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
    }
