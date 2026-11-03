"""Progress messages for slow steps: one line that updates in place.

    city_from_osm(1.2826, 103.8040, 500)
    # OpenStreetMap: asking overpass-api.de for 500 m around (1.2826, 103.8040)… 12 s
    # OpenStreetMap: 2,314 buildings, 812 roads and paths, ... in 18.2 s (1.9 MB)
    # city: [##########] 100% 32/32 strips in 1.1 s

Switch it off for one call with quiet=True, everywhere with
octreecity.progress.quiet(True), or with the environment variable
OCTREECITY_QUIET=1.

Where it goes: stderr (stdout stays clean). In a terminal or Jupyter, the
line is redrawn in place; elsewhere (a log file, CI) nothing is redrawn,
and a step prints one line when it ends. Steps shorter than DELAY seconds
print nothing. Messages also go to the logger "octreecity" (INFO).
A step that runs inside another one stays silent, so lines never mix.
"""

import logging
import os
import sys
import threading
import time

log = logging.getLogger("octreecity")

DELAY = 1.0          # seconds before a bar shows up
_quiet = os.environ.get("OCTREECITY_QUIET", "").strip().lower() in ("1", "true", "yes", "on")
_depth = 0           # steps running now (inner steps stay silent)
_lock = threading.Lock()
_last = [0]          # length of the line on screen


def quiet(on=True):
    """Switch every progress message off (on=True) or back on."""
    global _quiet
    _quiet = bool(on)


def is_quiet():
    return _quiet


def _silent(q):
    return bool(q) or _quiet or _depth > 0


def _stream():
    return getattr(sys, "stderr", None)


def _live():
    """True where a line can be redrawn: a terminal or a Jupyter kernel."""
    s = _stream()
    if s is None:
        return False
    if "ipykernel" in sys.modules:
        return True
    try:
        return bool(s.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _write(text, end):
    s = _stream()
    if s is None:
        return
    try:
        with _lock:
            if _live():
                pad = max(0, _last[0] - len(text))
                s.write("\r" + text + " " * pad + (end or ("\r" if not text else "")))
                _last[0] = 0 if end else len(text)
            elif end:
                s.write(text + end)
            s.flush()
    except (AttributeError, ValueError, OSError):
        pass                                  # no usable stderr: say nothing, never fail


def note(text, quiet=None):
    """A line of its own (a retry, a cache hit), kept on screen."""
    log.info(text)
    if not _silent(quiet):
        _write(text, "\n")


done = note


def size(n):
    """Bytes as text: 812 B, 1.4 MB."""
    n = float(n)
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def _secs(t):
    return f"{t:.1f} s" if t < 60 else f"{int(t // 60)} min {int(t % 60):02d} s"


class _Step:
    def __init__(self, quiet, delay):
        self.silent = _silent(quiet)
        self.delay = DELAY if delay is None else float(delay)
        self.t0 = time.perf_counter()
        self.shown = False

    @property
    def elapsed(self):
        return time.perf_counter() - self.t0

    def __enter__(self):
        global _depth
        _depth += 1
        return self

    def __exit__(self, *exc):
        global _depth
        _depth -= 1
        self.close(failed=exc[0] is not None)
        return False

    def _show(self, text):
        if self.silent:
            return
        if _live() and self.elapsed >= self.delay:
            _write(text, "")
            self.shown = True


class Bar(_Step):
    """A bar for a step with a known number of parts. Use as a context
    manager; call update(n) as parts finish."""

    def __init__(self, total, label, unit="", quiet=None, delay=None):
        super().__init__(quiet, delay)
        self.total, self.label, self.unit = max(int(total), 0), label, unit
        self.n = 0
        self._drawn = 0.0

    def update(self, n=1):
        self.n = min(self.total, self.n + n)
        now = time.perf_counter()
        if now - self._drawn >= 0.1 or self.n == self.total:
            self._drawn = now
            self._show(self.text())

    def text(self, final=False):
        f = self.n / self.total if self.total else 1.0
        bar = "#" * int(round(10 * f)) + "." * (10 - int(round(10 * f)))
        if self.unit == "bytes":
            n, total, unit = size(self.n), size(self.total), ""
        else:
            n, total, unit = f"{self.n:,}", f"{self.total:,}", f" {self.unit}" if self.unit else ""
        if final:
            return f"{self.label}: {n}{unit} in {_secs(self.elapsed)}"
        left = ""
        if 0 < self.n < self.total:
            left = f", {_secs(self.elapsed / self.n * (self.total - self.n))} left"
        return f"{self.label}: [{bar}] {f:4.0%} {n}/{total}{unit}{left}"

    def close(self, failed=False):
        if self.silent:
            return
        if failed:
            if self.shown:
                _write(f"{self.label}: stopped after {_secs(self.elapsed)}", "\n")
            return
        if self.elapsed >= self.delay:
            _write(self.text(final=True), "\n")
            log.info(self.text(final=True))


class Wait(_Step):
    """Elapsed time for a step with nothing to count (a server thinking,
    one long library call): the line ticks every half second from a
    background thread. set(text) changes the message; extra is shown
    after the time (e.g. bytes received)."""

    def __init__(self, text, quiet=None, delay=None, done=None):
        super().__init__(quiet, delay)
        self.message, self.extra, self.done = text, "", done
        self._stop = threading.Event()
        self._thread = None
        if not self.silent and not _live() and self.delay == 0:
            _write(text + "…", "\n")              # logs: say what we are waiting for, once

    def __enter__(self):
        super().__enter__()
        if not self.silent and _live():
            self._show(self.text())                  # at once when delay is 0
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        return self

    def set(self, text, extra=None):
        self.message = text
        if extra is not None:
            self.extra = extra

    def text(self):
        return f"{self.message}… {self.elapsed:.0f} s" + (f", {self.extra}" if self.extra else "")

    def _tick(self):
        while not self._stop.wait(0.5):
            self._show(self.text())

    def close(self, failed=False):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
        if self.silent:
            return
        if self.shown:
            _write("", "")                         # clear the line for what comes next
        if self.done and not failed and self.elapsed >= self.delay:
            _write(f"{self.done} in {_secs(self.elapsed)}", "\n")
            log.info(f"{self.done} in {_secs(self.elapsed)}")


def bar(total, label, unit="", quiet=None, delay=None):
    """with bar(len(parts), "octree", "levels", quiet) as b: ... b.update()"""
    return Bar(total, label, unit, quiet, delay)


def wait(text, quiet=None, delay=None, done=None):
    """with wait("links: finding pairs", quiet, done="links: pairs found"): slow_call()
    (done, if given, is printed with the time when the step took longer
    than delay)."""
    return Wait(text, quiet, delay, done)


def worker_quiet():
    """Called in worker processes, so they do not print over the parent's bar."""
    quiet(True)
