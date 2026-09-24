"""A small rounded capsule that drops from the top of the screen.

It is a heads-up indicator, not a window. It never takes focus - stealing focus
would break the very thing you are about to act on, because the window you were
typing into stops being the active one the moment the panel appears.

Rounded corners on Windows come from `-transparentcolor`: one exact colour is
punched out of the window, so a rounded rectangle drawn on that background has
genuinely round edges rather than a dark box with painted corners. The key
colour is a value nothing else uses, because anything drawn in it disappears.
"""
from __future__ import annotations

import queue
import tkinter as tk
from dataclasses import dataclass
from typing import Callable, Optional

KEY = "#010203"          # punched out of the window; never draw with it
BG = "#14161c"
EDGE = "#2b3140"
FG = "#e9edf4"
DIM = "#98a2b3"
ACCENT = "#6cc4ff"
GOOD = "#5ed68f"
BAD = "#ff6a6a"
WARN = "#f7bf54"

W, H = 360, 46          # capsule
PAD = 14
BARS = 22


def _round_rect(c: tk.Canvas, x0, y0, x1, y1, r, **kw):
    """A rounded rectangle as a smoothed polygon - tkinter has no native one."""
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return c.create_polygon(pts, smooth=True, **kw)


@dataclass
class Overlay:
    """One reusable window, shown and hidden rather than rebuilt each time."""

    on_cancel: Optional[Callable[[], None]] = None
    title: str = "Jev"
    # Two tools run at once and both drop an identical capsule from the top of
    # the screen. With nothing to tell them apart you cannot know which one is
    # listening, which is the single most important thing the panel exists to
    # say. Each gets its own accent and wears its own name.
    accent: str = ACCENT
    badge: str = ""

    def __post_init__(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=KEY)
        try:
            self.root.attributes("-transparentcolor", KEY)
        except tk.TclError:
            self.root.configure(bg=BG)          # no compositor: square, still works
        self.canvas = tk.Canvas(self.root, width=W, height=H, bg=KEY, highlightthickness=0)
        self.canvas.pack()
        self._levels = [0.0] * BARS
        self._q: queue.Queue = queue.Queue()
        self._status = "Listening"
        self._tone = self.accent
        self._text = ""
        self._visible = False
        self._pulse = 0
        self._pending_hide = None      # a scheduled hide, cancellable
        self.root.bind("<Escape>", lambda _e: self._cancel())
        self._tick()

    # -- geometry ---------------------------------------------------------
    def _place(self, width: int) -> None:
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"{width}x{H}+{(sw - width) // 2}+18")

    def _badge_and_text(self) -> str:
        return (self.badge + "  " if self.badge else "") + self._text

    def _width(self) -> int:
        """Grow just enough to fit a result, then shrink back."""
        if not self._text:
            return W
        est = 118 + int(len(self._badge_and_text()) * 6.6)
        return max(W, min(est, self.root.winfo_screenwidth() - 80))

    # -- public API (thread-safe; everything goes through the queue) -------
    def show(self, status: str = "Listening") -> None:
        self._q.put(("show", status, "", self.accent))

    def set_status(self, status: str, text: str = "", tone: str = "") -> None:
        self._q.put(("status", status, text, tone or self.accent))

    def set_level(self, level: float) -> None:
        self._q.put(("level", level, "", ""))

    def hide(self, after_ms: int = 0) -> None:
        self._q.put(("hide", after_ms, "", ""))

    def close(self) -> None:
        self._q.put(("close", "", "", ""))

    # -- internals --------------------------------------------------------
    def _cancel(self) -> None:
        if self.on_cancel:
            self.on_cancel()

    def _tick(self) -> None:
        try:
            while True:
                kind, a, b, c = self._q.get_nowait()
                if kind in ("show", "status"):
                    # A hide scheduled by the PREVIOUS utterance must not fire
                    # while this one is listening. That is what made continuous
                    # mode look like it had stopped: the capsule disappeared
                    # 2.6s in, while the microphone was still very much open.
                    if self._pending_hide is not None:
                        try:
                            self.root.after_cancel(self._pending_hide)
                        except Exception:
                            pass
                        self._pending_hide = None
                if kind == "show":
                    self._status, self._text, self._tone = a, "", c
                    self._levels = [0.0] * BARS
                    self._place(W)
                    self.root.deiconify()
                    self.root.attributes("-topmost", True)
                    self._visible = True
                elif kind == "status":
                    self._status, self._text = a, b
                    self._tone = c or self.accent
                    self._place(self._width())
                elif kind == "level":
                    self._levels.append(float(a))
                    del self._levels[:-BARS]
                elif kind == "hide":
                    if a:
                        self._pending_hide = self.root.after(int(a), self._do_hide)
                    else:
                        self._do_hide()
                elif kind == "close":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        if self._visible:
            self._pulse = (self._pulse + 1) % 60
            self._draw()
        self.root.after(33, self._tick)

    def _do_hide(self) -> None:
        self._visible = False
        self.root.withdraw()

    def _draw(self) -> None:
        c = self.canvas
        w = self._width()
        c.configure(width=w)
        c.delete("all")
        r = H // 2
        _round_rect(c, 1, 1, w - 1, H - 1, r, fill=BG, outline=EDGE)

        # breathing dot on the left
        import math
        amp = 0.5 + 0.5 * math.sin(self._pulse / 60 * 2 * math.pi)
        rad = 4 + 1.6 * amp
        c.create_oval(PAD + 6 - rad, H / 2 - rad, PAD + 6 + rad, H / 2 + rad,
                      fill=self._tone, outline="")

        x = PAD + 20
        if self.badge:
            c.create_text(x, H / 2, text=self.badge, anchor="w", fill=self._tone,
                          font=("Segoe UI Semibold", 9))
            x += 10 + c.bbox(c.find_all()[-1])[2] - c.bbox(c.find_all()[-1])[0]
        c.create_text(x, H / 2, text=self._status, anchor="w", fill=FG,
                      font=("Segoe UI Semibold", 10))
        x += 8 + c.bbox(c.find_all()[-1])[2] - c.bbox(c.find_all()[-1])[0]

        if self._text:
            c.create_text(x + 6, H / 2, text=self._text, anchor="w", fill=DIM,
                          font=("Segoe UI", 10))
        else:
            # compact meter, right-aligned inside the capsule
            right = w - PAD - 4
            step = 5.0
            bx = right - BARS * step
            peak = max(max(self._levels or [0.0]), 0.02)
            for lv in self._levels:
                bh = max(2.0, min(1.0, lv / peak) * (H - 22))
                c.create_rectangle(bx, H / 2 - bh / 2, bx + 2.4, H / 2 + bh / 2,
                                   fill=self._tone, outline="")
                bx += step

    def run(self) -> None:
        self.root.mainloop()
