"""Run the whole check suite. One command, one verdict.

`smoke.py` used to hand-roll its own fake session, and it grew the very bug it
was written to catch: it asserted that *something* was rendered, a crash renders
an error capsule, and so it printed SMOKE OK while every command raised
TypeError. The checks now live in `tests/`, where a rendered error is a failure,
and this file only runs them.

    python smoke.py            # everything
    python smoke.py -k sites    # one slice
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    code = subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q",
                            *sys.argv[1:]], cwd=str(ROOT))
    print("SMOKE OK - the real session path runs and every call matches its signature"
          if code == 0 else "SMOKE FAILED - see above")
    sys.exit(code)
