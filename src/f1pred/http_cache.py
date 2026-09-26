"""Polite, resumable HTTP client.

Every response is written to disk keyed by URL. Re-running an ingest costs
zero network requests for anything already fetched, which is what makes the
whole pipeline safe to ctrl-C and restart.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests

from . import config

log = logging.getLogger(__name__)


class RateLimitedSession:
    """A requests session that never exceeds a minimum interval between calls,
    caches every 200 response on disk, and backs off politely on 429."""

    def __init__(
        self,
        cache_dir: Path = config.HTTP_CACHE,
        min_interval: float = config.JOLPICA_MIN_INTERVAL,
        user_agent: str = config.USER_AGENT,
        hourly_limit: int = config.JOLPICA_HOURLY_LIMIT,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.hourly_limit = hourly_limit
        self._last_call = 0.0
        # Timestamps of real network calls in the last hour; cache hits don't count.
        # Persisted because jolpica's budget is per IP and outlives the process - an
        # empty window after a restart would overrun it immediately.
        self._window_path = cache_dir / "_rate_window.json"
        self._window: deque[float] = deque(self._load_window())
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.requests_made = 0
        self.cache_hits = 0
        self.throttle_waits = 0

    # -- pacing ------------------------------------------------------------
    def _load_window(self) -> list[float]:
        if not self._window_path.exists():
            return []
        try:
            stamps = json.loads(self._window_path.read_text())
        except (json.JSONDecodeError, OSError):
            return []
        cutoff = time.time() - 3600
        recent = sorted(t for t in stamps if isinstance(t, (int, float)) and t > cutoff)
        if recent:
            log.info(
                "Carrying over %d requests made in the last hour by earlier runs (budget %d/hr)",
                len(recent),
                self.hourly_limit,
            )
        return recent

    def _save_window(self) -> None:
        try:
            self._window_path.write_text(json.dumps(list(self._window)))
        except OSError:
            pass  # never let bookkeeping break an ingest

    def _respect_limits(self) -> None:
        """Block until sending another request is within both published limits.

        The per-second limit is a simple interval. The hourly one needs a
        sliding window: if we have already made `hourly_limit` calls in the
        last 3600s, wait until the oldest of them ages out. Reacting to a 429
        after the fact is not enough, because a short exponential backoff
        cannot clear an hour-long window - it just burns retries.
        """

        # Window uses wall clock so it can be persisted across runs; the
        # interval uses monotonic so a clock change cannot stall the client.
        def _prune() -> float:
            cutoff = time.time() - 3600
            while self._window and self._window[0] < cutoff:
                self._window.popleft()
            return cutoff

        cutoff = _prune()
        if len(self._window) >= self.hourly_limit:
            wait = max(1.0, self._window[0] - cutoff + 1.0)
            self.throttle_waits += 1
            log.warning(
                "Hourly budget reached (%d/hr). Pausing %.1f min - normal on a full "
                "backfill; everything already fetched is cached, so ctrl-C is safe.",
                self.hourly_limit,
                wait / 60,
            )
            time.sleep(wait)
            _prune()

        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    # -- cache -------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, url: str) -> dict[str, Any] | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)  # corrupt entry, refetch
            return None

    def _write_cache(self, url: str, payload: dict[str, Any]) -> None:
        self._cache_path(url).write_text(json.dumps(payload))

    # -- fetch -------------------------------------------------------------
    def get_json(self, url: str, *, max_retries: int = 6, refresh: bool = False) -> dict[str, Any]:
        """Fetch a URL, serving it from disk if we have it.

        `refresh` goes back to the network and overwrites the cached copy. It
        exists because a cached response is not always a finished one: asking
        for standings after a round that has not run yet returns an empty body
        with a 200, and caching that by URL means the answer stays empty for
        the rest of the season. Anything whose content can still change has to
        be able to say so.
        """
        cached = None if refresh else self._read_cache(url)
        if cached is not None:
            self.cache_hits += 1
            return cached

        backoff = 5.0
        for attempt in range(max_retries):
            self._respect_limits()

            self._last_call = time.monotonic()
            self._window.append(time.time())
            self._save_window()
            resp = self._session.get(url, timeout=30)
            self.requests_made += 1

            if resp.status_code == 200:
                payload = resp.json()
                self._write_cache(url, payload)
                return payload

            if resp.status_code == 429:
                # Trust Retry-After when present. Otherwise back off hard:
                # a 429 here usually means the hourly window, not the burst
                # one, and seconds of waiting will not clear that.
                wait = float(resp.headers.get("Retry-After", backoff))
                self.throttle_waits += 1

                # Self-tune: being throttled means our spacing is too tight for
                # what this IP is allowed right now, so slow down permanently
                # rather than hitting the same wall on every subsequent call.
                old_interval = self.min_interval
                self.min_interval = min(
                    self.min_interval * config.JOLPICA_BACKOFF_GROWTH, config.JOLPICA_MAX_INTERVAL
                )
                if self.min_interval > old_interval:
                    log.info(
                        "Throttled - easing request spacing %.2fs -> %.2fs",
                        old_interval,
                        self.min_interval,
                    )

                log.warning(
                    "Rate limited (attempt %d/%d) - sleeping %.0fs. Safe to ctrl-C; progress is cached.",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                backoff = min(backoff * 3, 900)  # 5 -> 15 -> 45 -> 135 -> 405
                continue

            if 500 <= resp.status_code < 600:
                log.warning("Server error %d on %s - retrying", resp.status_code, url)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            resp.raise_for_status()

        raise RuntimeError(f"Gave up on {url} after {max_retries} attempts")

    def paginate(
        self, url_template: str, path_to_list: tuple[str, ...], *, refresh: bool = False
    ) -> list[dict[str, Any]]:
        """Walk an Ergast-style paginated endpoint until every record is collected.

        ``url_template`` must contain ``{limit}`` and ``{offset}`` placeholders.
        ``path_to_list`` is the key path inside MRData to the list of records.
        ``refresh`` re-fetches every page rather than reading the disk cache.
        """
        limit = config.JOLPICA_PAGE_SIZE
        offset = 0
        collected: list[dict[str, Any]] = []

        while True:
            payload = self.get_json(url_template.format(limit=limit, offset=offset), refresh=refresh)
            mrdata = payload["MRData"]

            node: Any = mrdata
            for key in path_to_list:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            records = node if isinstance(node, list) else []
            collected.extend(records)

            total = int(mrdata.get("total", 0))
            offset += limit
            if offset >= total or not records:
                break

        return collected

    def stats(self) -> str:
        return (
            f"{self.requests_made} requests, {self.cache_hits} cache hits, "
            f"{self.throttle_waits} throttle waits"
        )
