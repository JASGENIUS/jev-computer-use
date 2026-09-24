"""One window for both tools: settings, start/stop, and what they are doing.

Everything used to live in six .bat files, so tuning anything meant editing a
batch script - which is why nothing ever got tuned, and why the launchers
quietly drifted apart until a test caught them. This reads and writes the same
`settings.json` the launchers do, so there is one source of truth rather than
four copies of the same flags.

    python dashboard.py

Deliberately Tkinter: it is already a dependency because the overlay uses it,
it starts instantly, and a settings panel that needs a web server and a build
step is a settings panel nobody opens.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jevflow import settings  # noqa: E402

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

BG = "#14161c"
PANEL = "#1b1e27"
FG = "#e9edf4"
DIM = "#98a2b3"
EDGE = "#2b3140"
GOOD = "#5ed68f"
BAD = "#ff6a6a"
TOOLS = {
    "jevflow": {"title": "JevFlow", "what": "dictation - speak, it types",
                "accent": "#6cc4ff"},
    "jcu": {"title": "JCU", "what": "computer use - speak, it acts",
            "accent": "#f7bf54"},
}

# Which settings get a control, in order, and how each one is presented.
FIELDS = {
    "jevflow": [
        ("hotkey", "Hotkey", "text", None),
        ("model", "Model", "choice", ["auto", "large-v3", "large-v3-turbo",
                                      "medium.en", "small.en", "base.en", "tiny.en"]),
        ("silence_hold_s", "Silence before it stops (s)", "number", (0.5, 8.0)),
        ("clean_speech", "Remove filler words", "bool", None),
        ("clean_aggressive", "Also remove basically / literally", "bool", None),
        ("hold_to_talk", "Hold the key instead of pressing", "bool", None),
        ("fix_homophones", "Fix there/their with Jev (network; slow on CPU)",
         "bool", None),
        ("fix_fillers", "Ask Jev before deleting like / just (network)",
         "bool", None),
    ],
    "jcu": [
        ("hotkey", "Hotkey", "text", None),
        ("model", "Model", "choice", ["auto", "small.en", "base.en", "medium.en",
                                      "large-v3", "tiny.en"]),
        ("silence_hold_s", "Silence before it stops (s)", "number", (0.5, 8.0)),
        ("stream", "Act while I am still talking", "bool", None),
        ("partial_every_s", "How often to re-listen (s)", "number", (0.15, 2.0)),
        ("clean_speech", "Remove filler words", "bool", None),
        ("wake", "Wake word", "choice", ["none", "transcript", "acoustic"]),
        ("wake_word", "Wake phrase", "text", None),
        ("dry_run", "Decide but do nothing (safe test)", "bool", None),
    ],
}


def running_pids(tool_title: str) -> list[int]:
    """Which copies of this tool are alive right now."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' or "
             "Name='python.exe'\" | Where-Object { $_.CommandLine -like "
             f"'*main.py*{tool_title}*' " + "} | ForEach-Object { $_.ProcessId }"],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception:
        return []


class Dashboard:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Jev - settings")
        self.root.configure(bg=BG)
        self.root.geometry("980x720")
        self.vars: dict[str, dict[str, tk.Variable]] = {}
        self.status: dict[str, tk.Label] = {}
        self._log_q: queue.Queue = queue.Queue()
        self._build()
        self._poll_status()
        self._drain_log()

    # -- layout ----------------------------------------------------------
    def _build(self) -> None:
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=18, pady=(16, 6))
        tk.Label(head, text="Jev", bg=BG, fg=FG,
                 font=("Segoe UI Semibold", 20)).pack(side="left")
        tk.Label(head, text="   settings for both tools, in one place",
                 bg=BG, fg=DIM, font=("Segoe UI", 10)).pack(side="left")

        cols = tk.Frame(self.root, bg=BG)
        cols.pack(fill="both", expand=True, padx=18, pady=6)
        for tool in TOOLS:
            self._tool_panel(cols, tool)

        tk.Label(self.root, text="  What they are doing", bg=BG, fg=DIM,
                 font=("Segoe UI Semibold", 10)).pack(fill="x", padx=18, pady=(8, 2))
        self.logbox = tk.Text(self.root, height=9, bg=PANEL, fg=DIM,
                              insertbackground=FG, relief="flat",
                              font=("Cascadia Mono", 9), wrap="none")
        self.logbox.pack(fill="both", expand=False, padx=18, pady=(0, 16))
        threading.Thread(target=self._tail_logs, daemon=True).start()

    def _tool_panel(self, parent: tk.Frame, tool: str) -> None:
        meta = TOOLS[tool]
        box = tk.Frame(parent, bg=PANEL, highlightbackground=EDGE,
                       highlightthickness=1)
        box.pack(side="left", fill="both", expand=True, padx=(0, 12))

        bar = tk.Frame(box, bg=PANEL)
        bar.pack(fill="x", padx=14, pady=(12, 2))
        tk.Label(bar, text=meta["title"], bg=PANEL, fg=meta["accent"],
                 font=("Segoe UI Semibold", 13)).pack(side="left")
        self.status[tool] = tk.Label(bar, text="checking", bg=PANEL, fg=DIM,
                                     font=("Segoe UI", 9))
        self.status[tool].pack(side="right")
        tk.Label(box, text=meta["what"], bg=PANEL, fg=DIM,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(0, 8))

        self.vars[tool] = {}
        current = settings.load()[tool]
        for key, label, kind, extra in FIELDS[tool]:
            self._field(box, tool, key, label, kind, extra, current[key])

        btns = tk.Frame(box, bg=PANEL)
        btns.pack(fill="x", padx=14, pady=14)
        for text, fn in (("Apply & restart", lambda t=tool: self._restart(t)),
                         ("Stop", lambda t=tool: self._stop(t))):
            tk.Button(btns, text=text, command=fn, bg=EDGE, fg=FG,
                      activebackground=meta["accent"], relief="flat",
                      font=("Segoe UI", 9), padx=12, pady=6).pack(side="left",
                                                                  padx=(0, 8))

    def _field(self, parent, tool, key, label, kind, extra, value) -> None:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=14, pady=3)
        if kind == "bool":
            var = tk.BooleanVar(value=bool(value))
            tk.Checkbutton(row, text=label, variable=var, bg=PANEL, fg=FG,
                           selectcolor=BG, activebackground=PANEL,
                           activeforeground=FG, relief="flat",
                           font=("Segoe UI", 9)).pack(anchor="w")
        else:
            tk.Label(row, text=label, bg=PANEL, fg=DIM, width=28, anchor="w",
                     font=("Segoe UI", 9)).pack(side="left")
            if kind == "choice":
                var = tk.StringVar(value=str(value))
                ttk.Combobox(row, textvariable=var, values=extra, width=16,
                             state="readonly").pack(side="left")
            elif kind == "number":
                var = tk.DoubleVar(value=float(value))
                tk.Spinbox(row, textvariable=var, from_=extra[0], to=extra[1],
                           increment=0.05, width=8, bg=BG, fg=FG,
                           relief="flat", buttonbackground=EDGE).pack(side="left")
            else:
                var = tk.StringVar(value=str(value))
                tk.Entry(row, textvariable=var, width=18, bg=BG, fg=FG,
                         insertbackground=FG, relief="flat").pack(side="left")
        self.vars[tool][key] = var

    # -- actions ---------------------------------------------------------
    def _save(self, tool: str) -> None:
        for key, var in self.vars[tool].items():
            try:
                settings.set_value(tool, key, var.get())
            except Exception as exc:
                self._log(f"  could not save {tool}.{key}: {exc}")
        settings.reload()

    def _stop(self, tool: str) -> None:
        title = TOOLS[tool]["title"]
        pids = running_pids(title)
        for pid in pids:
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                               capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except Exception:
                pass
        self._log(f"  stopped {title} ({len(pids)} process)" if pids
                  else f"  {title} was not running")

    def _restart(self, tool: str) -> None:
        """Save, stop, start. Settings only take effect on a restart, so doing
        all three behind one button is the honest thing - an Apply that quietly
        changed nothing until the next reboot would be a lie."""
        self._save(tool)
        self._stop(tool)
        args = settings.to_args(tool)
        try:
            subprocess.Popen([str(Path(sys.executable).with_name("pythonw.exe")),
                              str(ROOT / "main.py"), *args], cwd=str(ROOT),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._log(f"  started {TOOLS[tool]['title']}: {' '.join(args)}")
        except Exception as exc:
            self._log(f"  FAILED to start {TOOLS[tool]['title']}: {exc}")

    # -- status and log --------------------------------------------------
    def _poll_status(self) -> None:
        def check():
            for tool in TOOLS:
                n = len(running_pids(TOOLS[tool]["title"]))
                self._log_q.put(("status", tool, n))
        threading.Thread(target=check, daemon=True).start()
        self.root.after(4000, self._poll_status)

    def _log(self, line: str) -> None:
        self._log_q.put(("line", line, 0))

    def _tail_logs(self) -> None:
        """Follow both log files, so the panel shows what the tools are doing."""
        import time
        offsets: dict[Path, int] = {}
        while True:
            for path in sorted(LOG_DIR.glob("*.log")) if LOG_DIR.is_dir() else []:
                try:
                    size = path.stat().st_size
                    last = offsets.get(path, max(0, size - 2000))
                    if size < last:
                        last = 0                    # the file was rotated
                    if size > last:
                        with path.open("r", encoding="utf-8", errors="ignore") as fh:
                            fh.seek(last)
                            for line in fh.read().splitlines():
                                if line.strip():
                                    self._log_q.put(("line", line[:180], 0))
                        offsets[path] = size
                except Exception:
                    continue
            time.sleep(1.0)

    def _drain_log(self) -> None:
        try:
            while True:
                kind, a, b = self._log_q.get_nowait()
                if kind == "status":
                    lbl = self.status[a]
                    lbl.config(text="running" if b else "stopped",
                               fg=GOOD if b else BAD)
                else:
                    self.logbox.insert("end", a + "\n")
                    self.logbox.see("end")
                    if int(self.logbox.index("end-1c").split(".")[0]) > 400:
                        self.logbox.delete("1.0", "200.0")
        except queue.Empty:
            pass
        self.root.after(200, self._drain_log)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    Dashboard().run()
