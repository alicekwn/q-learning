from __future__ import annotations

"""Compatibility layer to reuse `streamlit_app.state*` from NiceGUI.

This module installs a lightweight stub `streamlit` module whose
`session_state` behaves like a dict but is backed by per-client storage
using NiceGUI's client context.

The goal is to:
- avoid modifying the existing Streamlit-specific state modules, and
- allow NiceGUI pages to call functions in `streamlit_app.state` and
  `streamlit_app.state_econ` with the same API.
"""

from collections.abc import MutableMapping
from typing import Any, Iterator
import sys
import types

from nicegui import ui


_CLIENT_STATE: dict[int, dict[str, Any]] = {}


class _SessionStateProxy(MutableMapping[str, Any]):
    """Per-client mapping that mimics `st.session_state`."""

    def _store(self) -> dict[str, Any]:
        client = ui.context.client
        if client is None:
            # Fallback to a global store if no client context is available
            return _CLIENT_STATE.setdefault(-1, {})
        return _CLIENT_STATE.setdefault(client.id, {})

    # Mapping interface
    def __getitem__(self, key: str) -> Any:
        return self._store()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store())

    def __len__(self) -> int:
        return len(self._store())

    # Common dict helpers
    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self._store().get(key, default)

    def clear(self) -> None:  # type: ignore[override]
        self._store().clear()

    # Attribute-style access (st.session_state.foo)
    def __getattr__(self, name: str) -> Any:
        try:
            return self._store()[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._store()[name] = value


def install_streamlit_stub() -> None:
    """Install a minimal `streamlit` stub if the real package is not needed.

    This is safe to call multiple times; the first call inserts the stub
    into `sys.modules` if no custom stub has been registered yet.
    """
    if "streamlit" in sys.modules:
        # Assume caller intentionally provided a real or custom module.
        return

    stub = types.ModuleType("streamlit")
    # Minimal `session_state` implementation
    stub.session_state = _SessionStateProxy()  # type: ignore[attr-defined]

    # Minimal `progress` implementation used in batch training; it returns an
    # object with a `.progress(float)` method that is a no-op.
    class _ProgressBar:
        def __init__(self) -> None:
            self.value = 0.0

        def progress(self, value: float) -> None:
            self.value = float(value)

    def progress(initial: float = 0.0) -> _ProgressBar:
        bar = _ProgressBar()
        bar.progress(initial)
        return bar

    stub.progress = progress  # type: ignore[attr-defined]

    sys.modules["streamlit"] = stub


