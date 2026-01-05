"""Lightweight client for the Render-hosted blockchain backend.

Uses simple requests with short timeouts and raises RuntimeError on non-200 responses.
Keep this dependency-free (outside of requests) so it can be dropped into the existing stack.
"""

import requests

BASE_URL = "https://fakensethfa.onrender.com"
DEFAULT_TIMEOUT = 15


def register_flagged_source(url: str, publisher: str = "FakeNewsDetector", timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Register a flagged source on-chain via the Render bridge."""
    payload = {"url": url, "publisher": publisher}
    resp = requests.post(f"{BASE_URL}/register", json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Register failed: {resp.status_code} {resp.text}")
    return resp.json()


def check_source_reputation(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Return metadata if URL exists in registry; otherwise {"known": False}."""
    resp = requests.get(f"{BASE_URL}/getNews", params={"url": url}, timeout=timeout)
    if resp.status_code == 200:
        data = resp.json()
        data["known"] = True
        return data
    if resp.status_code == 404:
        return {"known": False}
    raise RuntimeError(f"Lookup failed: {resp.status_code} {resp.text}")


def get_sources_by_publisher(publisher: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Fetch all URLs registered for a publisher. Returns dict with urls/count/status."""
    resp = requests.get(
        f"{BASE_URL}/getNewsByPublisher",
        params={"publisher": publisher},
        timeout=timeout,
    )
    if resp.status_code == 200:
        return resp.json()
    raise RuntimeError(f"Publisher lookup failed: {resp.status_code} {resp.text}")
