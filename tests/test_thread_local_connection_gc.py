"""Regression tests for SQLite connection retention under thread churn."""

import gc
import importlib
import os
from pathlib import Path
import queue
import sqlite3
import threading

import pytest


_CONNECTION_MODULES = (
    "mnemosyne.core.memory",
    "mnemosyne.core.beam",
)


def _database_fd_count(db_path: Path) -> int:
    """Count descriptors held for one SQLite database and its sidecars."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("open-descriptor assertions require /proc")

    prefix = str(db_path)
    count = 0
    for fd_path in fd_dir.iterdir():
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if target == prefix or target.startswith(f"{prefix}-"):
            count += 1
    return count


def _directory_database_fd_count(directory: Path) -> int:
    """Count descriptors held for SQLite files under one directory."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("open-descriptor assertions require /proc")

    prefix = f"{directory}{os.sep}"
    count = 0
    for fd_path in fd_dir.iterdir():
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if target.startswith(prefix):
            count += 1
    return count


def _open_on_new_thread(module, db_path: Path):
    errors = []

    def open_connection():
        try:
            module._get_connection(db_path)
        except Exception as exc:  # pragma: no cover - surfaced by the assertion
            errors.append(exc)

    thread = threading.Thread(target=open_connection)
    thread.start()
    thread.join()
    assert not errors


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_short_lived_thread_connections_keep_database_fds_bounded(
    module_name, tmp_path
):
    """Thread churn must not retain connections until process-wide FD exhaustion."""
    module = importlib.import_module(module_name)
    db_path = tmp_path / f"{module_name.rsplit('.', 1)[-1]}.db"

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(40):
            _open_on_new_thread(module, db_path)

        assert _database_fd_count(db_path) <= 20
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_connection_that_outlives_its_creating_thread_stays_usable(
    module_name, tmp_path
):
    """Cleanup must not invalidate a connection handed to another thread."""
    module = importlib.import_module(module_name)
    retained_path = tmp_path / "retained.db"
    churn_path = tmp_path / "churn.db"
    connections = queue.Queue()

    def open_retained_connection():
        connections.put(module._get_connection(retained_path))

    owner = threading.Thread(target=open_retained_connection)
    owner.start()
    owner.join()
    retained = connections.get_nowait()

    for _ in range(40):
        _open_on_new_thread(module, churn_path)

    assert retained.execute("SELECT 1").fetchone()[0] == 1
    retained.close()


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_live_thread_database_switches_keep_fds_bounded(module_name, tmp_path):
    """A persistent worker switching databases must also trigger cleanup."""
    module = importlib.import_module(module_name)
    ready = threading.Event()
    release = threading.Event()
    errors = []

    def churn_connections():
        try:
            for index in range(64):
                module._get_connection(tmp_path / f"database-{index}.db")
            ready.set()
            release.wait(timeout=5)
        except Exception as exc:  # pragma: no cover - surfaced by the assertion
            errors.append(exc)
            ready.set()

    gc_was_enabled = gc.isenabled()
    gc.disable()
    worker = threading.Thread(target=churn_connections)
    worker.start()
    try:
        assert ready.wait(timeout=5)
        assert not errors
        assert _directory_database_fd_count(tmp_path) <= 20
    finally:
        release.set()
        worker.join(timeout=5)
        if gc_was_enabled:
            gc.enable()
        gc.collect()


# ── Explicit release contract (mnemosyne#1163) ───────────────────────────────


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_close_thread_connection_releases_the_handle_deterministically(
    module_name, tmp_path
):
    """The owning thread releases its handle without waiting for collection."""
    if not Path("/proc/self/fd").exists():
        pytest.skip("open-descriptor assertions require /proc")
    module = importlib.import_module(module_name)
    db_path = tmp_path / "explicit-close.db"
    observed = queue.Queue()

    def open_use_and_close():
        module._get_connection(db_path).execute("SELECT 1").fetchone()
        before = _database_fd_count(db_path)
        closed = module.close_thread_connection(db_path)
        observed.put((before, closed, _database_fd_count(db_path)))

    worker = threading.Thread(target=open_use_and_close)
    worker.start()
    worker.join()
    before, closed, after = observed.get_nowait()

    assert before >= 1
    assert closed is True
    assert after == 0


@pytest.mark.parametrize("module_name", _CONNECTION_MODULES)
def test_connection_is_recreated_after_explicit_close(module_name, tmp_path):
    """A released thread-local must reconnect on next use, not stay poisoned."""
    module = importlib.import_module(module_name)
    db_path = tmp_path / "reopen.db"
    observed = queue.Queue()

    def close_then_reuse():
        first = module._get_connection(db_path)
        assert module.close_thread_connection(db_path) is True
        second = module._get_connection(db_path)
        observed.put(
            (second is not first, second.execute("SELECT 1").fetchone()[0])
        )

    worker = threading.Thread(target=close_then_reuse)
    worker.start()
    worker.join()

    reopened, value = observed.get_nowait()
    assert reopened is True
    assert value == 1


def test_close_thread_connection_refuses_the_main_thread(tmp_path):
    """The main-thread cache is process-lifetime and must opt in explicitly."""
    from mnemosyne.core.memory import _get_connection, close_thread_connection

    db_path = tmp_path / "main-thread.db"
    connection = _get_connection(db_path)

    assert close_thread_connection(db_path) is False
    assert connection.execute("SELECT 1").fetchone()[0] == 1

    assert close_thread_connection(db_path, allow_main_thread=True) is True
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_close_thread_connection_leaves_other_paths_cached(tmp_path):
    """Releasing one database must not displace another database's cache."""
    from mnemosyne.core.memory import _get_connection, close_thread_connection

    kept_path = tmp_path / "kept.db"
    other_path = tmp_path / "other.db"
    observed = queue.Queue()

    def close_unrelated_path():
        kept = _get_connection(kept_path)
        closed = close_thread_connection(other_path)
        observed.put((closed, kept.execute("SELECT 1").fetchone()[0]))
        close_thread_connection(kept_path)

    worker = threading.Thread(target=close_unrelated_path)
    worker.start()
    worker.join()

    closed, value = observed.get_nowait()
    assert closed is False
    assert value == 1


def test_retained_instance_alias_stays_usable_after_creator_thread_exits(tmp_path):
    """Owner lifetime is not creator-thread lifetime.

    BeamMemory keeps ``self.conn`` and hands the same handle to five sub-stores,
    and connections are opened with ``check_same_thread=False``. A retained
    instance must therefore survive its creating thread's exit, its own
    collection, and unrelated connection churn -- the difference between an
    explicit release by the owner and a release inferred from thread death.
    """
    from mnemosyne.core.beam import BeamMemory

    db_path = tmp_path / "retained-alias.db"
    churn_path = tmp_path / "churn.db"
    held = {}

    def create_and_retain():
        beam = BeamMemory(session_id="alias-test", db_path=db_path)
        held["beam"] = beam
        held["substores_share_conn"] = all(
            store.conn is beam.conn
            for store in (
                beam.annotations,
                beam.canonical,
                beam.media,
                beam.episodic_graph,
                beam.veracity_consolidator,
            )
        )

    creator = threading.Thread(target=create_and_retain)
    creator.start()
    creator.join()
    assert not creator.is_alive()

    beam_module = importlib.import_module("mnemosyne.core.beam")
    gc.collect()
    for _ in range(8):
        _open_on_new_thread(beam_module, churn_path)

    assert held["substores_share_conn"] is True
    retained = held["beam"]
    assert retained.conn.execute("SELECT 1").fetchone()[0] == 1
    assert retained.annotations.conn.execute("SELECT 1").fetchone()[0] == 1


class _CloseFailingConnection(sqlite3.Connection):
    """Connection whose close can be made to fail on demand."""

    fail_close = True

    def close(self):
        if type(self).fail_close:
            raise sqlite3.OperationalError("simulated close failure")
        return super().close()


def test_failed_close_keeps_the_handle_cached_for_retry(tmp_path):
    """A failed close must not drop the only reference to a live handle.

    ``sqlite3.Connection.close()`` can raise, and the handle is still usable
    when it does. Clearing the thread-local anyway would hide a live handle and
    leave the caller no way to retry, so the entry must survive until a later
    release succeeds.
    """
    from mnemosyne.core import memory as memory_module

    db_path = (tmp_path / "close-failure.db").resolve()
    observed = queue.Queue()

    def close_with_failure():
        beam_module = importlib.import_module("mnemosyne.core.beam")
        connection = sqlite3.connect(str(db_path), factory=_CloseFailingConnection)
        memory_module._thread_local.conn = connection
        memory_module._thread_local.db_path = str(db_path)
        try:
            first = memory_module.close_thread_connection(db_path)
            still_cached = memory_module._thread_local.conn is connection
            usable = connection.execute("SELECT 1").fetchone()[0]

            _CloseFailingConnection.fail_close = False
            second = memory_module.close_thread_connection(db_path)
            cleared = memory_module._thread_local.conn is None
            observed.put((first, still_cached, usable, second, cleared))
        finally:
            _CloseFailingConnection.fail_close = False
            beam_module._thread_local.conn = None
            beam_module._thread_local.db_path = None

    worker = threading.Thread(target=close_with_failure)
    worker.start()
    worker.join()

    assert observed.get_nowait() == (False, True, 1, True, True)
