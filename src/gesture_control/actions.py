"""Map gesture events to OS actions through a swappable backend."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

from gesture_control.config import Config
from gesture_control.engine import Event

log = logging.getLogger(__name__)


class Backend(Protocol):
    def press(self, keys: tuple[str, ...]) -> None: ...
    def hotkey(self, keys: tuple[str, ...]) -> None: ...
    def scroll(self, clicks: int) -> None: ...
    def move_to(self, x: float, y: float) -> None: ...  # normalized 0..1 screen coordinates
    def mouse_down(self) -> None: ...
    def mouse_up(self) -> None: ...
    def click(self) -> None: ...
    def launch(self, targets: tuple[str, ...]) -> str | None: ...


@dataclass
class RecordingBackend:
    """Records calls instead of touching the OS. Used by tests and `--dry-run`."""

    calls: list[tuple[str, Any]] = field(default_factory=list)

    def press(self, keys: tuple[str, ...]) -> None:
        self.calls.append(("press", keys))

    def hotkey(self, keys: tuple[str, ...]) -> None:
        self.calls.append(("hotkey", keys))

    def scroll(self, clicks: int) -> None:
        self.calls.append(("scroll", clicks))

    def move_to(self, x: float, y: float) -> None:
        self.calls.append(("move_to", (x, y)))

    def mouse_down(self) -> None:
        self.calls.append(("mouse_down", None))

    def mouse_up(self) -> None:
        self.calls.append(("mouse_up", None))

    def click(self) -> None:
        self.calls.append(("click", None))

    def launch(self, targets: tuple[str, ...]) -> str | None:
        self.calls.append(("launch", targets))
        return targets[0] if targets else None


def open_target(target: str) -> None:
    """Open an app path, URI (onenote:) or shell:AppsFolder id. Raises OSError on failure."""
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", target], check=True)
    else:
        subprocess.run(["xdg-open", target], check=True)


class PyAutoGuiBackend:
    """Real keyboard and mouse control. pyautogui is imported lazily (it needs a display)."""

    def __init__(self) -> None:
        import pyautogui

        pyautogui.PAUSE = 0  # no built-in sleep after each call; the engine already debounces
        # The corner fail-safe would abort when the gesture cursor reaches a screen corner.
        # Pause from the tray or press `p` in the overlay instead.
        pyautogui.FAILSAFE = False
        self._pg = pyautogui
        self._w, self._h = pyautogui.size()

    def press(self, keys: tuple[str, ...]) -> None:
        for key in keys:
            self._pg.press(key)

    def hotkey(self, keys: tuple[str, ...]) -> None:
        self._pg.hotkey(*keys)

    def scroll(self, clicks: int) -> None:
        self._pg.scroll(clicks * 120)  # one wheel notch on Windows is 120 units

    def move_to(self, x: float, y: float) -> None:
        self._pg.moveTo(int(x * (self._w - 1)), int(y * (self._h - 1)))

    def mouse_down(self) -> None:
        self._pg.mouseDown()

    def mouse_up(self) -> None:
        self._pg.mouseUp()

    def click(self) -> None:
        self._pg.click()

    def launch(self, targets: tuple[str, ...]) -> str | None:
        for target in targets:
            try:
                open_target(target)
                return target
            except (OSError, subprocess.CalledProcessError) as exc:
                log.info("could not open %s: %s", target, exc)
        log.warning("none of the launch targets opened: %s", targets)
        return None


class ActionDispatcher:
    """Looks up the action bound to each event and runs it on the backend."""

    def __init__(self, config: Config, backend: Backend) -> None:
        self.cfg = config
        self.backend = backend

    def action_for(self, event: Event) -> str | None:
        binding = self.cfg.gestures.get(event.name)
        return binding.action if binding else None

    def dispatch(self, event: Event) -> str | None:
        """Run the event's action. Returns the action name, or None if the event is unbound."""
        name = self.action_for(event)
        if name is None:
            return None
        spec = self.cfg.actions[name]
        b = self.backend
        match spec.type:
            case "key":
                b.press(spec.keys)
            case "hotkey":
                b.hotkey(spec.keys)
            case "launch":
                b.launch(spec.targets)
            case "scroll":
                b.scroll(int(event.value or 0))
            case "move_cursor":
                if event.value is not None:
                    b.move_to(*event.value)
            case "mouse_down":
                b.mouse_down()
            case "mouse_up":
                b.mouse_up()
            case "click":
                b.click()
        return name

    def label(self, action: str) -> str:
        spec = self.cfg.actions.get(action)
        return (spec.label or action) if spec else action
