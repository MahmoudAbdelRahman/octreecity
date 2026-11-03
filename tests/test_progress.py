"""Progress messages: shown when wanted, silent when not, never a crash."""

import importlib
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import shapely

from octreecity import easy, progress
from octreecity.builder import CityBuilder
from octreecity.features import Features

OSM = {"elements": [
    {"type": "way", "id": 1, "tags": {"building": "yes", "height": "12"},
     "geometry": [{"lat": 1.2820, "lon": 103.8030}, {"lat": 1.2820, "lon": 103.8032},
                  {"lat": 1.2822, "lon": 103.8032}, {"lat": 1.2822, "lon": 103.8030},
                  {"lat": 1.2820, "lon": 103.8030}]},
    {"type": "way", "id": 2, "tags": {"highway": "residential"},
     "geometry": [{"lat": 1.2815, "lon": 103.8025}, {"lat": 1.2830, "lon": 103.8040}]},
    {"type": "node", "id": 3, "tags": {"natural": "tree"}, "lat": 1.2824, "lon": 103.8036},
]}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(progress, "_quiet", False)
    monkeypatch.setattr(progress, "DELAY", 0.0)
    monkeypatch.setattr(progress, "_depth", 0)
    yield


def live(monkeypatch, on=True):
    monkeypatch.setattr(progress, "_live", lambda: on)


def small_city(quiet=None):
    b = CityBuilder(0, 0, 64, 64, 1.0)
    b.add_buildings(Features([shapely.box(10, 10, 20, 20)], height=[12.0]))
    return b.build(air_above=4, quiet=quiet)


# ------------------------------------------------------------- switching off
def test_quiet_everywhere(capsys, monkeypatch):
    small_city(quiet=True)
    assert capsys.readouterr().err == ""
    progress.quiet(True)
    small_city()
    small_city().to_octree(3)
    progress.note("hello")
    assert capsys.readouterr().err == ""
    progress.quiet(False)
    small_city()
    assert "city 64 x 64 m at 1 m: 4 strips in" in capsys.readouterr().err


def test_environment_variable(monkeypatch, capsys):
    monkeypatch.setenv("OCTREECITY_QUIET", "1")
    importlib.reload(progress)
    try:
        assert progress.is_quiet()
        progress.note("hello")
        assert capsys.readouterr().err == ""
    finally:
        monkeypatch.delenv("OCTREECITY_QUIET")
        importlib.reload(progress)
    assert not progress.is_quiet()


def test_short_steps_print_nothing(capsys, monkeypatch):
    monkeypatch.setattr(progress, "DELAY", 60.0)
    small_city().to_octree(3)
    assert capsys.readouterr().err == ""


# ------------------------------------------------------------- where it goes
def test_no_terminal_one_line_per_step(capsys):
    small_city().to_octree(3)
    err = capsys.readouterr().err
    assert "\r" not in err                                   # pytest's stderr is no terminal
    lines = err.splitlines()
    assert lines[0].startswith("city 64 x 64 m at 1 m: 4 strips in")
    assert lines[1].startswith("octree: 5 levels in")
    assert len(lines) == 2


def test_terminal_redraws_one_line(capsys, monkeypatch):
    live(monkeypatch)
    with progress.bar(3, "work", "parts") as b:
        for _ in range(3):
            time.sleep(0.11)
            b.update()
    err = capsys.readouterr().err
    assert "\rwork: [" in err and "3/3 parts" in err
    assert err.endswith("\n") and err.rstrip().split("\r")[-1].startswith("work: 3 parts in")


def test_wait_ticks_then_clears(capsys, monkeypatch):
    live(monkeypatch)
    with progress.wait("asking", done="answered") as w:
        time.sleep(0.7)
        w.extra = "1.0 kB"
    err = capsys.readouterr().err
    assert "\rasking… 0 s, " in err or "\rasking… 1 s" in err
    assert err.rstrip().split("\r")[-1].startswith("answered in")


def test_no_stderr_never_fails(monkeypatch):
    monkeypatch.setattr(sys, "stderr", None)
    small_city().to_octree(3)
    with progress.wait("x", delay=0):
        pass
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(sys, "stderr", closed)
    live(monkeypatch)
    with progress.wait("x", delay=0):
        time.sleep(0.6)
    small_city()
    progress.note("hello")


def test_inner_steps_stay_silent(capsys):
    with progress.bar(1, "outer", "parts") as b:
        small_city().to_octree(3)
        progress.note("inner note")
        b.update()
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1 and lines[0].startswith("outer: 1 parts in")


def test_bar_failure_does_not_print_success(capsys, monkeypatch):
    live(monkeypatch)
    with pytest.raises(RuntimeError):
        with progress.bar(2, "work") as b:
            b.update()
            raise RuntimeError("boom")
    err = capsys.readouterr().err
    assert "stopped after" in err and "in 0" not in err.split("stopped")[-1]
    assert progress._depth == 0


# ------------------------------------------------------------- OpenStreetMap
class _Overpass(BaseHTTPRequestHandler):
    mode = "ok"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.mode == "busy":
            self.send_response(429)
            self.end_headers()
            return
        time.sleep(0.3)                                      # the server thinking
        body = json.dumps(OSM).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()                                   # no length: like Overpass
        for k in range(0, len(body), 100):
            self.wfile.write(body[k:k + 100])
            self.wfile.flush()

    def log_message(self, *a):
        pass


def _server(mode):
    handler = type(f"H{mode}", (_Overpass,), {"mode": mode})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/api/interpreter"


@pytest.fixture
def overpass(monkeypatch):
    busy, busy_url = _server("busy")
    ok, ok_url = _server("ok")
    monkeypatch.setattr(easy, "OVERPASS", [busy_url, ok_url])
    yield busy_url, ok_url
    busy.shutdown()
    ok.shutdown()


def test_osm_download_messages(capsys, monkeypatch, overpass, tmp_path):
    busy_url, ok_url = overpass
    host_busy, host_ok = busy_url.split("/")[2], ok_url.split("/")[2]
    out = easy.fetch_osm(1.2826, 103.8040, 300, cache=tmp_path / "osm.json", timeout=5)
    assert out == OSM
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == f"OpenStreetMap: asking {host_busy} for 300 m around (1.2826, 103.8040)…"
    assert lines[1] == f"OpenStreetMap: {host_busy} busy (HTTP 429), trying {host_ok}…"
    assert lines[2] == f"OpenStreetMap: asking {host_ok} for 300 m around (1.2826, 103.8040)…"
    assert lines[3].startswith("OpenStreetMap: 1 buildings, 1 roads and paths, 1 trees in ")
    assert "kB" in lines[3] or " B" in lines[3]
    easy.fetch_osm(1.2826, 103.8040, 300, cache=tmp_path / "osm.json")
    assert "using the saved answer" in capsys.readouterr().err


def test_osm_download_live(capsys, monkeypatch, overpass):
    live(monkeypatch)
    _, ok_url = overpass
    monkeypatch.setattr(easy, "OVERPASS", [ok_url])
    easy.fetch_osm(1.2826, 103.8040, 300, timeout=5)
    err = capsys.readouterr().err
    assert "OpenStreetMap: asking 127.0.0.1:" in err
    assert err.rstrip().split("\r")[-1].startswith("OpenStreetMap: 1 buildings")


def test_osm_download_quiet(capsys, overpass):
    easy.fetch_osm(1.2826, 103.8040, 300, timeout=5, quiet=True)
    assert capsys.readouterr().err == ""


def test_osm_every_server_fails(capsys, monkeypatch, overpass):
    busy_url, _ = overpass
    monkeypatch.setattr(easy, "OVERPASS", [busy_url])
    monkeypatch.setattr(easy.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="all Overpass servers failed"):
        easy.fetch_osm(1.2826, 103.8040, 300, timeout=5)
    err = capsys.readouterr().err
    assert "every server failed, trying again in 5 s" in err
    assert err.rstrip().endswith("every server failed")


def test_city_from_osm_quiet_and_not(capsys):
    easy.city_from_osm(1.2826, 103.8040, 200, osm_json=OSM, quiet=True)
    assert capsys.readouterr().err == ""
    easy.city_from_osm(1.2826, 103.8040, 200, osm_json=OSM)
    assert capsys.readouterr().err.startswith("city 200 x 200 m at 1 m:")


def test_summary_counts():
    assert easy._osm_summary(OSM) == "1 buildings, 1 roads and paths, 1 trees"
    assert easy._osm_summary({"elements": []}) == "nothing found"
    assert np.isfinite(len(easy._why(TimeoutError("timed out"))))
