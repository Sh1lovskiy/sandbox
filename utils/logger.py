# utils/logger.py
"""Logging helpers built on top of loguru."""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, TypeVar, cast

from loguru import logger as _logger
from loguru._logger import Logger as LoguruLogger
from tqdm.auto import tqdm

T = TypeVar("T")


# ============================== CONFIG =======================================

MODULE_W = 7
LINE_W = 3


@dataclass(frozen=True)
class LoggingCfg:
    """Project-wide logging configuration."""

    level: str = "INFO"
    json: bool = True
    log_dir: Path = Path(".logs")
    log_format: str = (
        "<green>{time:MM-DD HH:mm:ss}</green>"
        "[<level>{level:.3}</level>]"
        f"[<cyan>{{extra[module]:<{MODULE_W}.{MODULE_W}}}</cyan>:"
        f"<cyan>{{line:>{LINE_W}}}</cyan>] "
        "<level>{message}</level>"
    )
    # file logs are JSON (serialize=True), so this format is unused in practice.
    log_file_format: str = (
        f"{{time:YYYY-MM-DD HH:mm:ss}}[{{level:.3}}]"
        f"[{{extra[module]:<{MODULE_W}.{MODULE_W}}}:{{line:>{LINE_W}}}] "
        "{{message}}"
    )
    progress_bar_format: str = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )


# Exposed default config instance
LOGCFG = LoggingCfg()


# ============================== LOGGER =======================================


class Logger:
    """Thin wrapper around loguru with unified configuration."""

    _configured: bool = False
    _log_dir: Path = LOGCFG.log_dir
    _log_file: Optional[Path] = None
    _lock = threading.Lock()

    @staticmethod
    def _add_sinks(level: str, json_format: bool) -> None:
        """Attach console and file sinks."""
        os.makedirs(Logger._log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        Logger._log_file = Logger._log_dir / f"{ts}.log.json"

        _logger.add(
            sys.stdout,
            level=level,
            serialize=False,
            format=LOGCFG.log_format,
        )
        _logger.add(
            Logger._log_file,
            level=level,
            serialize=json_format,
            format=LOGCFG.log_file_format,
        )

    @staticmethod
    def _configure(level: str, json_format: bool) -> None:
        """Configure sinks once (thread-safe)."""
        with Logger._lock:
            if Logger._configured:
                return
            _logger.remove()
            Logger._add_sinks(level, json_format)
            Logger._configured = True

    @staticmethod
    def configure(
        level: Optional[str] = None,
        log_dir: Optional[Path | str] = None,
        json_format: Optional[bool] = None,
    ) -> None:
        """
        Manually configure the logger.
        Call before any get_logger() to take effect.
        """
        if log_dir is not None:
            Logger._log_dir = Path(log_dir)
        lvl = level or LOGCFG.level
        jsn = LOGCFG.json if json_format is None else bool(json_format)
        Logger._configure(lvl, jsn)

    @staticmethod
    def configure_root_logger(level: str = "WARNING") -> None:
        """Simplify third-party library logs by resetting root sinks."""
        with Logger._lock:
            _logger.remove()
            _logger.add(sys.stdout, level=level)
            Logger._configured = True

    @staticmethod
    def get_logger(
        name: str,
        level: Optional[str] = None,
        json_format: Optional[bool] = None,
    ) -> LoguruLogger:
        """
        Return a configured loguru logger bound to ``name`` (in extra[module]).
        """
        Logger._configure(
            level or LOGCFG.level,
            LOGCFG.json if json_format is None else bool(json_format),
        )
        return _logger.bind(module=name)

    @staticmethod
    def progress(
        iterable: Iterable[T],
        desc: Optional[str] = None,
        total: Optional[int] = None,
    ) -> Iterable[T]:
        """Unified tqdm wrapper with project bar style."""
        return cast(
            Iterable[T],
            tqdm(
                iterable,
                desc=desc,
                total=total,
                leave=False,
                bar_format=LOGCFG.progress_bar_format,
            ),
        )


# ============================== CONTEXTS =====================================


class CaptureStderrToLogger:
    """Redirect native (C/C++) stderr (fd=2) lines into the logger."""

    def __init__(self, logger: LoguruLogger):
        self.logger = logger
        self.pipe_read: Optional[int] = None
        self.pipe_write: Optional[int] = None
        self.thread: Optional[threading.Thread] = None
        self._old_stderr_fd: Optional[int] = None

    def _reader(self) -> None:
        assert self.pipe_read is not None
        with os.fdopen(self.pipe_read, "r", errors="replace") as fh:
            for line in fh:
                msg = line.rstrip()
                if msg:
                    self.logger.warning(f"[STDERR] {msg}")

    def __enter__(self) -> "CaptureStderrToLogger":
        self._old_stderr_fd = os.dup(2)
        self.pipe_read, self.pipe_write = os.pipe()
        os.dup2(self.pipe_write, 2)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        if self._old_stderr_fd is not None:
            os.dup2(self._old_stderr_fd, 2)
        if self.pipe_write is not None:
            os.close(self.pipe_write)
        if self._old_stderr_fd is not None:
            os.close(self._old_stderr_fd)
        if self.thread is not None:
            self.thread.join(timeout=0.2)


class SuppressO3DInfo:
    """Silence noisy stdout/stderr from libs (e.g., Open3D)."""

    def __init__(self) -> None:
        self._old_stdout: Optional[int] = None
        self._old_stderr: Optional[int] = None
        self._devnull: Optional[int] = None

    def __enter__(self) -> "SuppressO3DInfo":
        self._old_stdout = os.dup(1)
        self._old_stderr = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._old_stdout is not None:
            os.dup2(self._old_stdout, 1)
            os.close(self._old_stdout)
        if self._old_stderr is not None:
            os.dup2(self._old_stderr, 2)
            os.close(self._old_stderr)
        if self._devnull is not None:
            os.close(self._devnull)
