"""The mining pipeline's only route to a model: maitred on the LAN, with the
knoll key, restricted to LAN-served models.

The gateway enforces the same allowlist on the knoll key (the NixOS maitred
module); this is the client-side half, so a typo or a changed default cannot
send session content toward a hosted model. The gateway address and key file
are deliberately not configurable.
"""

import json

from ..endpoint import Endpoint, chat_reply

GATEWAY = "http://10.0.0.2:4002"
KEY_FILE = "/run/agenix/maitred-keys"
KEY_VAR = "MAITRED_KEY_KNOLL"
LOCAL_MODELS = frozenset(
    {"radiance/qwen3.8-27b", "gufo/qwen3.8-flash-next", "unholy/dsv4-flash", "embed/bge-m3"}
)


def knoll_key():
    with open(KEY_FILE) as handle:
        for line in handle:
            name, _, value = line.strip().partition("=")
            if name.removeprefix("export ").strip() == KEY_VAR:
                return value.strip().strip("'\"")
    raise SystemExit("%s is not in %s" % (KEY_VAR, KEY_FILE))


class Model:
    def __init__(self, name, timeout=1800, retries=3):
        if name not in LOCAL_MODELS:
            raise SystemExit("refusing model %s: not LAN-served" % name)
        self.name = name
        self.endpoint = Endpoint(GATEWAY, timeout, retries, api_key=knoll_key())

    def ask(self, prompt, generation):
        """One user turn; returns `chat_reply`'s dict plus `prompt_tokens`."""
        response = self.endpoint.chat(self.name, prompt, generation)
        reply = chat_reply(response)
        reply["prompt_tokens"] = (response.get("usage") or {}).get("prompt_tokens")
        return reply

    def ask_json(self, prompt, generation):
        """The first JSON object in the answer, retried once if it is missing
        or cut off. Returns (object or None, prompt tokens, completion tokens)."""
        used = [0, 0]
        for attempt in range(2):
            reply = self.ask(prompt if attempt == 0 else prompt + "\nAnswer with the JSON object only.", generation)
            used[0] += reply["prompt_tokens"] or 0
            used[1] += reply["completion_tokens"] or 0
            if reply["finish_reason"] == "length":
                continue
            text = reply["text"]
            start, end = text.find("{"), text.rfind("}")
            try:
                found = json.loads(text[start : end + 1]) if 0 <= start < end else None
            except ValueError:
                found = None
            if isinstance(found, dict):
                return found, used[0], used[1]
        return None, used[0], used[1]

    def embed(self, texts):
        """One vector per text, in order."""
        response = self.endpoint.post("/v1/embeddings", {"model": self.name, "input": list(texts)})
        return [item["embedding"] for item in sorted(response["data"], key=lambda item: item["index"])]
