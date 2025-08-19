"""Centralized unhandled exception tracking."""

from __future__ import annotations

import os
import signal
import sys
import traceback
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from .logger import Logger


class CameraError(Exception):
    """Base class for camera related errors."""


class CameraConnectionError(CameraError):
    """Raised when the camera device cannot be opened."""


class ErrorTracker:
    """Installable global exception hook that logs uncaught errors."""

    logger = Logger.get_logger("utils.error_tracker")
    _installed = False
    _orig_hook: Optional[Callable[..., None]] = None
    _cleanup_funcs: List[Callable[[], None]] = []
    _keyboard_listener: Any = None
    _terminal_echo: Any = None

    @classmethod
    def register_cleanup(cls, func: Callable[[], None]) -> None:
        """Register a cleanup function executed on fatal errors."""
        cls._cleanup_funcs.append(func)

    @classmethod
    def _run_cleanup(cls) -> None:
        """Execute all registered cleanup callbacks."""

        for func in cls._cleanup_funcs:
            try:
                func()
            except Exception as e:
                cls.logger.error(f"Cleanup failed: {e}")
        cls.stop_keyboard_listener()

    @classmethod
    def install_excepthook(cls) -> None:
        """Log unhandled exceptions through the project logger."""
        if cls._installed:
            return

        cls._orig_hook = sys.excepthook

        def _hook(exc_type, exc, tb) -> None:
            message = "".join(traceback.format_exception(exc_type, exc, tb))
            cls.logger.error(f"Unhandled exception:\n{message}")
            cls._run_cleanup()
            if cls._orig_hook:
                cls._orig_hook(exc_type, exc, tb)

        sys.excepthook = _hook
        cls._installed = True
        cls.logger.debug("Global exception hook installed")

    @classmethod
    def install_signal_handlers(cls) -> None:
        """Shutdown gracefully on SIGINT or SIGTERM."""

        def _handler(signum, frame) -> None:
            cls.logger.info(f"Received signal {signum}")
            cls._run_cleanup()
            raise SystemExit(1)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    @classmethod
    def install_keyboard_listener(cls, stop_key: str = "esc") -> None:
        """Start a background listener that exits on ``stop_key`` or ``Ctrl+C``."""

        if cls._keyboard_listener is not None:
            return
        try:
            from .keyboard import GlobalKeyListener, TerminalEchoSuppressor
        except Exception as e:
            cls.logger.warning(f"Keyboard listener unavailable: {e}")
            return

        def _on_stop() -> None:
            """Handle hotkey press by running cleanup and exiting."""
            cls.logger.info(f"Stop key {stop_key} pressed")
            cls._run_cleanup()
            os._exit(1)

        hotkeys = {f"<{stop_key}>": _on_stop, "<ctrl>+c": _on_stop}
        cls._keyboard_listener = GlobalKeyListener(hotkeys)
        cls._keyboard_listener.start()
        cls._terminal_echo = TerminalEchoSuppressor()
        cls._terminal_echo.start()
        cls.logger.debug(f"Keyboard listener installed for keys: {list(hotkeys)}")

    @classmethod
    def stop_keyboard_listener(cls) -> None:
        """Stop the background keyboard listener if running."""

        if cls._keyboard_listener is not None:
            try:
                cls._keyboard_listener.stop()
            finally:
                cls._keyboard_listener = None
                cls.logger.debug("Keyboard listener stopped")
        if cls._terminal_echo is not None:
            try:
                cls._terminal_echo.stop()
            finally:
                cls._terminal_echo = None

    @classmethod
    def report(cls, exc: Exception) -> None:
        """Log an exception with full traceback to logger only."""
        tb = exc.__traceback__
        if tb:
            formatted = "".join(traceback.format_exception(type(exc), exc, tb))
        else:
            stack = "".join(traceback.format_stack())
            formatted = f"{type(exc).__name__}: {exc}\nTraceback (most recent call last):\n{stack}"
        cls.logger.error(formatted)
