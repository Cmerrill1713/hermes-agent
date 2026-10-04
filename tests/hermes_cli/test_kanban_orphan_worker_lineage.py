"""A LIVE worker orphaned by a gateway crash must be reclaimed, not held forever.

Regression test for the claim deadlock: ``_worker_alive`` only checks liveness +
PID fingerprint, so a worker reparented to launchd (PPID 1) by a crashed
dispatcher kept its ``running`` claim indefinitely. With
``max_concurrent_workers: 1`` nothing else could ever start.
"""
import os
import signal
import time

import psutil
import pytest


def _make_orphaned_worker():
    """child -> intermediate -> grandchild; kill the intermediate so the
    grandchild is reparented to launchd while still alive.

    The grandchild's PID is handed back over a pipe. NEVER rediscover it by
    scanning the host for ``ppid == 1`` + ``python``: the host has other such
    processes (searxng, the RLM gateway, Home Assistant, ComfyUI, and any
    leaked pytest run), and a first-match scan signals whichever sorts first --
    this test would SIGKILL an unrelated service. Only PIDs this function
    created are ever signalled.
    """
    read_fd, write_fd = os.pipe()
    me = os.getpid()
    inter = os.fork()
    if inter == 0:
        os.close(read_fd)
        grandchild = os.fork()
        if grandchild == 0:
            # Orphan-in-waiting: closes no fd it can write to and just sleeps.
            while True:
                time.sleep(0.2)
        os.write(write_fd, str(grandchild).encode())
        os.close(write_fd)
        while True:
            time.sleep(0.2)
    os.close(write_fd)
    # Read EXACTLY ONCE. The writer closes its end right after writing, so a single
    # read() returns the payload. Looping until a newline would never terminate:
    # the writer sends no newline and the forked children inherit the read end, so
    # the pipe never reaches EOF.
    raw = os.read(read_fd, 64)
    os.close(read_fd)
    pid = int(raw.split(b"\n")[0].strip() or b"0")
    assert pid > 1, f"grandchild pid handoff failed (read {raw!r})"
    time.sleep(0.6)
    os.kill(inter, signal.SIGKILL)
    time.sleep(0.9)
    assert psutil.Process(pid).ppid() == 1, (
        "grandchild was not reparented to launchd -- the orphan path this test "
        "exercises would not be reached"
    )
    return pid


@pytest.mark.live_system_guard_bypass
def test_live_orphan_is_reclaimed():
    """Real signals to real forked children we created ourselves are the subject
    under test, so the live-system guard is bypassed deliberately. The forked
    processes belong to this test process; no gateway or user process is touched."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as D

    kb.init_db()
    conn = kbc.connect()
    pid = _make_orphaned_worker()
    me = os.getpid()
    fp = D._process_fingerprint(pid)
    lock = kb._claimer_id().split(":", 1)[0] + ":" + str(me + 100000)
    conn.execute(
        "INSERT INTO tasks (id,title,status,assignee,worker_pid,claim_lock,"
        "claim_expires,worker_started_at,current_run_id,created_at) "
        "VALUES ('t_orphan','orphan card','running','default',?,?,?,?,1,?)",
        (pid, lock, time.time() + 3600, fp, time.time() - 600))
    conn.commit()

    D.dispatch_once(conn, spawn_fn=lambda *a, **k: None)

    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id='t_orphan' ORDER BY created_at")]
    row = conn.execute("SELECT claim_lock,consecutive_failures,worker_pid "
                       "FROM tasks WHERE id='t_orphan'").fetchone()
    assert "orphaned_worker" in kinds, "live orphan must be reclaimed"
    assert row["consecutive_failures"] == 0, "gateway crash is not a task failure"
    assert row["worker_pid"] is None
    if psutil.pid_exists(pid):
        os.kill(pid, signal.SIGKILL)


def test_legitimate_live_worker_is_never_reaped():
    from hermes_cli import kanban_db_dispatch as D
    pid = os.fork()
    if pid == 0:
        while True:
            time.sleep(0.2)
    time.sleep(0.3)
    try:
        # claim_lock names the REAL parent -> not orphaned
        assert D._worker_orphaned(pid, "host:%d" % os.getpid()) is False
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def test_orphan_predicate_fails_open():
    from hermes_cli import kanban_db_dispatch as D
    assert D._worker_orphaned(None, "host:1") is False
    assert D._worker_orphaned(4242, None) is False
    assert D._worker_orphaned(4242, "no-colon") is False
    assert D._worker_orphaned(4242, "host:1") is False


@pytest.mark.live_system_guard_bypass
def test_live_orphan_that_is_still_heartbeating_is_not_reaped():
    """A gateway restart reparents workers to launchd while they keep working.

    Keying only on parent lineage would kill healthy in-flight work on every
    restart, so an orphan is reclaimed only once it has gone quiet for longer
    than TERMINAL_WORKER_REAP_GRACE_SECONDS.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as D

    kb.init_db()
    conn = kbc.connect()
    pid = _make_orphaned_worker()
    me = os.getpid()
    fp = D._process_fingerprint(pid)
    lock = kb._claimer_id().split(":", 1)[0] + ":" + str(me + 100000)
    conn.execute(
        "INSERT INTO tasks (id,title,status,assignee,worker_pid,claim_lock,"
        "claim_expires,worker_started_at,current_run_id,created_at,last_heartbeat_at) "
        "VALUES ('t_live','live orphan','running','default',?,?,?,?,1,?,?)",
        (pid, lock, time.time() + 3600, fp, time.time() - 600, int(time.time()) - 5))
    conn.commit()

    D.dispatch_once(conn, spawn_fn=lambda *a, **k: None)

    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id='t_live' ORDER BY created_at")]
    assert "orphaned_worker" not in kinds, "a heartbeating orphan is doing live work"
    assert psutil.pid_exists(pid), "live orphan worker must not be terminated"
    if not psutil.pid_exists(pid):
        os.kill(pid, signal.SIGKILL)
