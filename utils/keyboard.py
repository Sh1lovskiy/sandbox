# utils/keyboard.py
"""Global hotkey utilities used by CLI tools."""

from __future__ import annotations

import sys
import termios
from typing import Callable, Dict, List, Optional

from pynput import keyboard
from .logger import Logger

KeyAction = Callable[[], None]


class GlobalKeyListener:
    """Listen for global hotkeys and suppress terminal output."""

    def __init__(
        self, hotkeys: Dict[str, KeyAction], *, suppress: bool = False
    ) -> None:
        """Set up hotkey callbacks and configure pynput listener."""

        self.logger = Logger.get_logger("keyboard")
        self.hotkeys = hotkeys
        self.listener = keyboard.GlobalHotKeys(hotkeys, suppress=suppress)
        self.listener.daemon = True

    def start(self) -> None:
        """Start listening for configured hotkeys."""
        self.listener.start()
        self.logger.debug(
            f"GlobalKeyListener started with keys: {list(self.hotkeys)}"
        )

    def stop(self) -> None:
        """Stop the hotkey listener."""
        try:
            self.listener.stop()
        finally:
            self.logger.debug("GlobalKeyListener stopped")


class TerminalEchoSuppressor:
    """Disable terminal echo to hide typed hotkeys."""

    def __init__(self) -> None:
        """Cache terminal settings for later restoration."""

        self.logger = Logger.get_logger("utils.terminal")
        self.fd = sys.stdin.fileno()
        self.enabled = False
        self._orig_attrs: Optional[List[int]] = None

    def start(self) -> None:
        """Disable echo if running in a TTY."""
        if self.enabled or not sys.stdin.isatty():
            return
        try:
            self._orig_attrs = termios.tcgetattr(self.fd)
            new_attrs = termios.tcgetattr(self.fd)
            new_attrs[3] &= ~termios.ECHO
            termios.tcsetattr(self.fd, termios.TCSADRAIN, new_attrs)
            self.enabled = True
            self.logger.debug("Terminal echo disabled")
        except Exception as e:
            self.logger.error(f"Failed to disable terminal echo: {e}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def stop(self) -> None:
        """Restore echo settings."""
        if not self.enabled or self._orig_attrs is None:
            return
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._orig_attrs)
        finally:
            self.enabled = False
            self.logger.debug("Terminal echo restored")
