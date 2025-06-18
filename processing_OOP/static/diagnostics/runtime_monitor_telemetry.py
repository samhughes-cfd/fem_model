# processing_OOP/static/diagnostics/runtime_monitor_telemetry.py

"""
Transparent, lightweight resource telemetry.

* machine header  ➜  diagnostics/RuntimeMonitorTelemetry.log
* live snapshots  ➜  one line per call to `sample()`  **or**
                   via the `with monitor.stage("XYZ"):` helper.
"""

from __future__ import annotations
import os, sys, shutil, platform, datetime, subprocess, psutil, logging
from pathlib import Path
from typing import Dict, Sequence
try:
    import cpuinfo                     # optional
except ImportError:
    cpuinfo = None


class RuntimeMonitorTelemetry:
    HEADER_WIDTH = 74
    CPU_INTERVAL = 0.2        # seconds for cpu_percent sample

    # ─────────────────────────────────────────────────────────────────── init
    def __init__(
        self,
        job_results_dir: str | Path,
        filename: str = "RuntimeMonitorTelemetry.log",
    ) -> None:
        self.job_root = Path(job_results_dir).resolve().parent
        self.diag_dir = self.job_root / "diagnostics"
        self.diag_dir.mkdir(parents=True, exist_ok=True)

        self.path     = self.diag_dir / filename
        self.logger   = self._init_logger()

        # Header only once per file
        if self.path.stat().st_size == 0:
            self._write_machine_header()

        # warm-up so the first value isn’t 0 %
        psutil.cpu_percent(None)
        self._last_ts: datetime.datetime | None = None   # for Δt

    # ────────────────────────────────────────────────────────────── context
    class _StageTimer:
        """Internal context-manager to auto-emit BEGIN/END rows."""
        def __init__(self, monitor: "RuntimeMonitorTelemetry", label: str):
            self.m = monitor
            self.label = label
        def __enter__(self):
            self.m.sample(f"{self.label} BEGIN")
        def __exit__(self, exc_type, exc, tb):
            self.m.sample(f"{self.label} END")

    def stage(self, label: str) -> "_StageTimer":
        """`with monitor.stage("AssembleGlobal"):` convenience."""
        return self._StageTimer(self, label)

    # ────────────────────────────────────────────────────────────── public
    def sample(self, label: str) -> Dict[str, float]:
        """
        Append a live snapshot (RAM, CPU, disk) tagged with *label* and
        return the numbers so callers may reuse them.
        """
        snap = self._current_usage()
        now  = datetime.datetime.now()
        delta = (now - self._last_ts).total_seconds() if self._last_ts else 0.0
        self._last_ts = now

        self.logger.info(
            "%s │ %-26s │ Δt %5.2f s │ RAM %7.1f MB │ CPU %5.1f %% │ Disk %6.1f GB",
            now.strftime("%Y-%m-%d %H:%M:%S"),
            label,
            delta,
            snap["RAM_MB"], snap["CPU_%"], snap["Disk_GB"]
        )
        return snap

    # ───────────────────────────────────────────────────────── internal
    def _init_logger(self) -> logging.Logger:
        lg = logging.getLogger(f"RuntimeMonitorTelemetry.{id(self)}")
        lg.handlers.clear()
        lg.setLevel(logging.INFO)
        lg.propagate = False

        fh = logging.FileHandler(self.path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(fh)
        return lg

    # .......................................................... header
    def _write_machine_header(self) -> None:
        p   = platform
        ram = psutil.virtual_memory().total / 2**30
        d_total, _, d_free = shutil.disk_usage("/")
        cpu_name = (
            cpuinfo.get_cpu_info().get("brand_raw") if cpuinfo else p.processor()
        )

        lines: Sequence[str] = [
            "-" * self.HEADER_WIDTH,
            f"Machine specifications – written {datetime.datetime.now()}",
            f"OS          : {p.system()} {p.release()} ({p.version()})",
            f"CPU         : {cpu_name}",
            f"   logical  : {psutil.cpu_count(logical=True)}",
            f"   physical : {psutil.cpu_count(logical=False)}",
            f"RAM         : {ram:.2f} GB",
            f"Disk        : {d_total/2**30:.2f} GB total  /  {d_free/2**30:.2f} GB free",
            f"Python      : {p.python_version()}  ({sys.executable})",
        ]

        # (best-effort) GPU on Windows
        if p.system().lower() == "windows":
            try:
                raw = subprocess.check_output(
                    "wmic path win32_VideoController get name",
                    shell=True,
                    stderr=subprocess.DEVNULL,
                ).decode(errors="ignore").splitlines()
                gpus = [ln.strip() for ln in raw[1:] if ln.strip()]
                lines += [f"GPU(s)      : {', '.join(gpus) or '—'}"]
            except Exception:                       # noqa: BLE001
                lines += ["GPU(s)      : <unavailable>"]

        lines += ["-" * self.HEADER_WIDTH]

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n\n")

        self.logger.info("Runtime telemetry header written → %s", self.path)

    # .......................................................... snapshot
    @classmethod
    def _current_usage(cls) -> Dict[str, float]:
        proc = psutil.Process(os.getpid())
        return {
            "RAM_MB" : proc.memory_info().rss / 2**20,
            "Disk_GB": psutil.disk_usage('/').used / 2**30,
            "CPU_%"  : psutil.cpu_percent(interval=cls.CPU_INTERVAL),
        }