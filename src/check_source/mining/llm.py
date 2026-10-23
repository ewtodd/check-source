"""The mining pipeline's only route to a model: maitred on the LAN, with the
knoll key, restricted to LAN-served models.

The gateway enforces the same allowlist on the knoll key (the NixOS maitred
module); this is the client-side half, so a typo or a changed default cannot
send session content toward a hosted model. The gateway address and key file
are deliberately not configurable.
"""

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
