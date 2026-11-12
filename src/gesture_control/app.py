"""Command-line entry point: live app (tray + overlay), recorder, replayer, config template."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

from gesture_control.actions import ActionDispatcher, Backend, RecordingBackend
from gesture_control.classifier import Pose
from gesture_control.config import Config, default_config_text, load_config
from gesture_control.engine import FrameResult, GestureEngine

log = logging.getLogger("gesture_control")
WINDOW = "Gesture control"


class Controller:
    """Shared state between the camera loop and the tray icon."""

    def __init__(self, config: Config, backend: Backend, show_overlay: bool = True) -> None:
        self.config = config
        self.engine = GestureEngine(config)
        self.dispatcher = ActionDispatcher(config, backend)
        self.paused = False
        self.overlay_visible = show_overlay
        self.stop = threading.Event()
        self.on_change: list = []  # callbacks run when paused/overlay change (tray refresh)

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            for event in self.engine.reset(time.monotonic()):
                self.dispatcher.dispatch(event)
        self._changed()

    def set_overlay(self, visible: bool) -> None:
        self.overlay_visible = visible
        self._changed()

    def _changed(self) -> None:
        for cb in self.on_change:
            cb()

    def run_camera_loop(self) -> None:
        import cv2

        from gesture_control.detector import HandDetector
        from gesture_control.overlay import OverlayState, draw

        cam = self.config.camera
        backend_flag = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(cam.index, backend_flag)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
        if not cap.isOpened():
            log.error("could not open camera %s", cam.index)
            self.stop.set()
            return

        detector = HandDetector(self.config.model)
        state = OverlayState()
        window_open = False
        t0 = time.monotonic()
        last = t0
        try:
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                if cam.mirror:
                    frame = cv2.flip(frame, 1)
                now = time.monotonic()
                t = now - t0
                state.fps = 0.9 * state.fps + 0.1 / max(now - last, 1e-3)
                last = now
                hand = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t)

                state.paused = self.paused
                if self.paused:
                    result = FrameResult(t, hand, Pose.NONE, Pose.NONE, False)
                else:
                    result = self.engine.process(hand, t)
                    self._dispatch(result, state)

                if self.overlay_visible:
                    draw(frame, result, state)
                    cv2.imshow(WINDOW, frame)
                    window_open = True
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.stop.set()
                    elif key == ord("p"):
                        self.set_paused(not self.paused)
                    elif key == ord("h"):
                        self.set_overlay(False)
                    elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        self.set_overlay(False)  # closed with the window's X button
                elif window_open:
                    cv2.destroyWindow(WINDOW)
                    cv2.waitKey(1)
                    window_open = False
        finally:
            for event in self.engine.reset(time.monotonic() - t0):
                self.dispatcher.dispatch(event)
            detector.close()
            cap.release()
            cv2.destroyAllWindows()

    def _dispatch(self, result: FrameResult, state) -> None:
        for event in result.events:
            try:
                action = self.dispatcher.dispatch(event)
            except Exception:  # an OS call failing must not kill the camera loop
                log.exception("action for %s failed", event.name)
                continue
            if event.name == "mouse_mode":
                state.last_action = "Mouse mode on" if event.value else "Mouse mode off"
                state.last_action_t = event.t
            elif action and event.name not in ("cursor", "scroll"):
                state.last_action = self.dispatcher.label(action)
                state.last_action_t = event.t
                log.info("%s -> %s", event.name, action)


def run_tray(controller: Controller) -> None:
    import pystray
    from PIL import Image, ImageDraw

    def icon_image(paused: bool):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((4, 4, 60, 60), fill=(150, 150, 150) if paused else (60, 170, 90))
        for i, x in enumerate((18, 27, 36, 45)):  # four fingers
            d.rounded_rectangle((x - 3, 14 + (i % 3), x + 3, 36), 3, fill="white")
        d.rounded_rectangle((16, 30, 48, 50), 6, fill="white")
        return img

    icon = pystray.Icon(
        "gesture-control",
        icon_image(False),
        "Gesture control",
        menu=pystray.Menu(
            pystray.MenuItem(
                "Pause gestures",
                lambda: controller.set_paused(not controller.paused),
                checked=lambda _: controller.paused,
            ),
            pystray.MenuItem(
                "Show camera overlay",
                lambda: controller.set_overlay(not controller.overlay_visible),
                checked=lambda _: controller.overlay_visible,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: controller.stop.set()),
        ),
    )

    def refresh() -> None:
        icon.icon = icon_image(controller.paused)
        icon.update_menu()

    controller.on_change.append(refresh)
    worker = threading.Thread(target=controller.run_camera_loop, daemon=True)
    worker.start()

    def watch_stop() -> None:
        controller.stop.wait()
        worker.join(timeout=3)
        icon.stop()

    threading.Thread(target=watch_stop, daemon=True).start()
    icon.run()


# -- subcommands ------------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.dry_run:
        backend: Backend = RecordingBackend()
        log.info("dry run: actions are logged, not performed")
    else:
        from gesture_control.actions import PyAutoGuiBackend

        backend = PyAutoGuiBackend()
    controller = Controller(config, backend, show_overlay=not args.no_overlay)
    if args.no_tray:
        controller.overlay_visible = True
        controller.run_camera_loop()
    else:
        run_tray(controller)


def cmd_record(args: argparse.Namespace) -> None:
    """Record landmarks from the webcam into a fixture file (no actions are performed)."""
    import cv2

    from gesture_control.detector import HandDetector
    from gesture_control.fixtures import Recording

    config = load_config(args.config)
    cam = config.camera
    cap = cv2.VideoCapture(cam.index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    detector = HandDetector(config.model)
    frames = []
    t0 = time.monotonic()
    start = t0 + args.countdown
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if cam.mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            hand = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), now - t0)
            if now >= start:
                frames.append((now - start, hand))
                label, colour = f"recording {now - start:4.1f}/{args.seconds}s", (0, 0, 255)
            else:
                label, colour = f"get ready: {start - now:3.1f}", (0, 200, 255)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.imshow("Record", frame)
            if cv2.waitKey(1) & 0xFF == ord("q") or now - start >= args.seconds:
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()

    out = Path(args.out or f"recordings/{args.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    expected = {"actions": args.expect} if args.expect is not None else {}
    Recording(args.name, frames, args.description, "recorded", expected).save(out)
    seen = sum(h is not None for _, h in frames)
    print(f"saved {len(frames)} frames ({seen} with a hand) to {out}")


def cmd_replay(args: argparse.Namespace) -> None:
    from gesture_control.fixtures import load_recording

    config = load_config(args.config)
    rec = load_recording(args.file)
    engine = GestureEngine(config)
    dispatcher = ActionDispatcher(config, RecordingBackend())
    previous = None
    for t, hand in rec.frames:
        result = engine.process(hand, t)
        if result.pose != previous:
            print(f"{t:6.2f}s  pose  {result.pose.value}")
            previous = result.pose
        for event in result.events:
            if event.name == "cursor" and not args.verbose:
                continue
            action = dispatcher.dispatch(event)
            value = f" {event.value}" if event.value is not None else ""
            print(f"{t:6.2f}s  event {event.name}{value} -> {action or '-'}")


def cmd_init_config(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists; pass --force to overwrite")
    path.write_text(default_config_text(), "utf-8")
    print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gesture-control", description=__doc__)
    p.add_argument("-c", "--config", help="YAML config (default: built-in)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command")

    run = sub.add_parser("run", help="start the app (default)")
    for target in (p, run):
        target.add_argument("--dry-run", action="store_true", help="log actions, don't do them")
        target.add_argument("--no-tray", action="store_true", help="overlay window only")
        target.add_argument("--no-overlay", action="store_true", help="start with overlay hidden")
    run.set_defaults(func=cmd_run)

    rec = sub.add_parser("record", help="record landmarks from the webcam into a fixture")
    rec.add_argument("name")
    rec.add_argument("--seconds", type=float, default=3.0)
    rec.add_argument("--countdown", type=float, default=3.0)
    rec.add_argument("--out", help="output path (default: recordings/NAME.json)")
    rec.add_argument("--description", default="")
    rec.add_argument(
        "--expect", nargs="*", metavar="ACTION", help="expected actions, for use as a test"
    )
    rec.set_defaults(func=cmd_record)

    rep = sub.add_parser("replay", help="run a recording through the engine and print events")
    rep.add_argument("file")
    rep.set_defaults(func=cmd_replay)

    init = sub.add_parser("init-config", help="write the default config to a file to edit")
    init.add_argument("path", nargs="?", default="gestures.yaml")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init_config)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    func = getattr(args, "func", cmd_run)
    func(args)


if __name__ == "__main__":
    main()
