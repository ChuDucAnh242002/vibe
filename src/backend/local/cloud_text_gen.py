import json
import os
from typing import Iterator
from urllib.parse import urlsplit

import requests

from abstract_classes import TextGenerationInterface


class GatewayError(RuntimeError):
    pass


class GatewayTextGenService(TextGenerationInterface):
    """Send explicitly approved text requests to the authenticated gateway."""

    def __init__(self):
        self.url = os.environ["VIBE_GATEWAY_URL"].rstrip("/")
        self.token = os.environ["VIBE_GATEWAY_TOKEN"]
        parsed = urlsplit(self.url)
        if (
            not self.token
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.scheme not in ("http", "https")
            or not parsed.hostname
        ):
            raise ValueError("A valid gateway URL and nonempty token are required")
        if parsed.scheme == "http" and parsed.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
            "gateway",
        ):
            raise ValueError("Remote gateways require HTTPS")

    def generate(self, user_input: str, context: str = "") -> Iterator[str]:
        try:
            yield from self._generate(user_input, context)
        except (requests.RequestException, ValueError, RuntimeError) as error:
            raise GatewayError(
                "Gateway request failed; check connectivity, authentication, and gateway logs"
            ) from error

    def _generate(self, user_input: str, context: str) -> Iterator[str]:
        with requests.post(
            self.url + "/generate",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"prompt": user_input, "context": context},
            stream=True,
            timeout=(5, 35),
            allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Gateway rejected the request (HTTP {response.status_code})"
                )
            for line in response.iter_lines(chunk_size=1):
                if not line:
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise RuntimeError("Invalid gateway event")
                if event.get("error"):
                    raise RuntimeError(
                        "Cloud generation failed; response may be incomplete"
                    )
                if event.get("done") is True:
                    return
                text = event.get("text")
                if not isinstance(text, str):
                    raise RuntimeError("Gateway event is missing text")
                yield text
        raise RuntimeError("Gateway disconnected before completing the response")
