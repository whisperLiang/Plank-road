from __future__ import annotations

import threading
from contextlib import contextmanager
from collections.abc import Iterator


_TORCHLENS_FORWARD_LOCK = threading.RLock()


@contextmanager
def torchlens_forward_guard() -> Iterator[None]:
    """Serialize TorchLens tracing with background model forwards.

    TorchLens keeps active logging state at process scope, so unrelated model
    forwards from other threads can corrupt an in-flight trace. Code paths that
    call TorchLens prepare/logging and background teacher inference share this
    guard.
    """

    with _TORCHLENS_FORWARD_LOCK:
        yield

