"""
client.py — Typed OpenEnv client for DataCleaningEnv
=====================================================
Usage (sync — recommended for notebooks and scripts)
-----------------------------------------------------
    from client import DataCleaningClient

    with DataCleaningClient(base_url="http://localhost:8000").sync() as client:
        obs = client.reset(task_level="medium")
        print(obs)

        result = client.step(action=1)       # impute_missing
        print(result.reward.value, result.reward.done)

        state = client.state()
        print(state.score, state.remaining_issues)

Usage (async)
-------------
    import asyncio
    from client import DataCleaningClient

    async def main():
        async with DataCleaningClient(base_url="http://localhost:8000") as client:
            obs = await client.reset(task_level="hard")
            result = await client.step(action=3)   # fix_outlier
            state  = await client.state()

    asyncio.run(main())

Action IDs
----------
    0  →  skip           (do nothing, move to next issue)
    1  →  impute_missing  (fill NaN with column mean / mode)
    3  →  fix_outlier     (replace outlier with column mean)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

import httpx

from models import (
    ActionModel,
    EnvStateModel,
    ObservationModel,
    StepResultModel,
)

# ---------------------------------------------------------------------------
# Default server URL — change if deployed elsewhere
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------
class DataCleaningClient:
    """
    Async OpenEnv client for DataCleaningEnv.

    Each instance owns a unique session_id so multiple clients can
    run against the same server without interfering.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        session_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id or str(uuid.uuid4())
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ── headers ──────────────────────────────────────────────────────────
    @property
    def _headers(self) -> dict[str, str]:
        return {"X-Session-Id": self.session_id}

    # ── lifecycle ─────────────────────────────────────────────────────────
    async def __aenter__(self) -> "DataCleaningClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_open(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "Client is not open. Use 'async with DataCleaningClient(...) as client:'"
            )
        return self._client

    # ── API methods ───────────────────────────────────────────────────────
    async def reset(self, task_level: str = "medium") -> Optional[ObservationModel]:
        """
        Start a new episode.

        Parameters
        ----------
        task_level : "easy" | "medium" | "hard"

        Returns
        -------
        ObservationModel with the first row to inspect, or None if
        the dataset happens to be already clean.
        """
        http = self._ensure_open()
        resp = await http.post("/reset", json={"task_level": task_level})
        resp.raise_for_status()
        data = resp.json()
        obs_data = data.get("observation")
        if obs_data is None:
            return None
        return ObservationModel(**obs_data)

    async def step(self, action: int) -> StepResultModel:
        """
        Take one action in the environment.

        Parameters
        ----------
        action : int
            0 = skip | 1 = impute_missing | 3 = fix_outlier

        Returns
        -------
        StepResultModel containing the next observation, reward, done flag,
        and step count.

        Raises
        ------
        ValueError  if action is not one of {0, 1, 3}.
        httpx.HTTPStatusError  if the server returns an error (e.g. 400 if
            /reset has not been called yet).
        """
        # Validate locally before hitting the network
        ActionModel(action=action)  # raises ValueError on bad action

        http = self._ensure_open()
        resp = await http.post("/step", json={"action": action})
        resp.raise_for_status()
        return StepResultModel(**resp.json())

    async def state(self) -> EnvStateModel:
        """
        Fetch the full current environment state without advancing the episode.

        Returns
        -------
        EnvStateModel with step count, remaining issues, score, episode log, etc.
        """
        http = self._ensure_open()
        resp = await http.get("/state")
        resp.raise_for_status()
        return EnvStateModel(**resp.json())

    # ── sync wrapper ──────────────────────────────────────────────────────
    def sync(self) -> "SyncDataCleaningClient":
        """Return a synchronous wrapper around this async client."""
        return SyncDataCleaningClient(
            base_url=self.base_url,
            session_id=self.session_id,
            timeout=self.timeout,
        )


# ---------------------------------------------------------------------------
# Sync wrapper — uses httpx.Client internally (no asyncio required)
# ---------------------------------------------------------------------------
class SyncDataCleaningClient:
    """
    Synchronous OpenEnv client for DataCleaningEnv.

    Mirrors the async DataCleaningClient interface exactly,
    but uses blocking httpx calls.

    Preferred for:
    - Jupyter notebooks
    - Training scripts that don't use asyncio
    - Quick experiments

    Example
    -------
        with DataCleaningClient("http://localhost:8000").sync() as client:
            obs   = client.reset("hard")
            done  = False
            while not done:
                result = client.step(action=1)
                done   = result.reward.done
            state = client.state()
            print("Score:", state.score)
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        session_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id or str(uuid.uuid4())
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-Session-Id": self.session_id}

    # ── lifecycle ─────────────────────────────────────────────────────────
    def __enter__(self) -> "SyncDataCleaningClient":
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self.timeout,
        )
        return self

    def __exit__(self, *_) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def _ensure_open(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError(
                "Client is not open. Use 'with client.sync() as c:'"
            )
        return self._client

    # ── API methods ───────────────────────────────────────────────────────
    def reset(self, task_level: str = "medium") -> Optional[ObservationModel]:
        """Start a new episode. Returns initial ObservationModel or None."""
        http = self._ensure_open()
        resp = http.post("/reset", json={"task_level": task_level})
        resp.raise_for_status()
        data = resp.json()
        obs_data = data.get("observation")
        if obs_data is None:
            return None
        return ObservationModel(**obs_data)

    def step(self, action: int) -> StepResultModel:
        """Take one action. Returns StepResultModel."""
        ActionModel(action=action)  # local validation

        http = self._ensure_open()
        resp = http.post("/step", json={"action": action})
        resp.raise_for_status()
        return StepResultModel(**resp.json())

    def state(self) -> EnvStateModel:
        """Fetch full env state without advancing the episode."""
        http = self._ensure_open()
        resp = http.get("/state")
        resp.raise_for_status()
        return EnvStateModel(**resp.json())
