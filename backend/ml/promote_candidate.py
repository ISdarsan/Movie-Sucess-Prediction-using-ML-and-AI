"""Promote the candidate model to be the live website model — only if its integration tests pass.

    .venv/Scripts/python -m backend.ml.promote_candidate

Steps: (1) run backend/tests/test_candidate.py against the candidate; abort on any failure. (2) make sure the model that was live
is preserved as pre_release_movie_classifier.v1_lightgbm.joblib (rollback). (3) copy candidate -> live model and live schema.
(4) delete the candidate files. The previous live files are never deleted.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "backend" / "artifacts"
CAND, CSCHEMA = ROOT / "pre_release_movie_classifier.candidate.joblib", ART / "prerelease_schema.candidate.json"
LIVE, LSCHEMA = ROOT / "pre_release_movie_classifier.joblib", ART / "prerelease_schema.json"
BACKUP, BSCHEMA = ROOT / "pre_release_movie_classifier.v1_lightgbm.joblib", ART / "prerelease_schema_v1.json"


def main() -> int:
    if not (CAND.exists() and CSCHEMA.exists()):
        print("no candidate to promote"); return 1
    r = subprocess.run([sys.executable, "-m", "pytest", "backend/tests/test_candidate.py", "-q", "-p", "no:cacheprovider"], cwd=ROOT)
    if r.returncode != 0:
        print("\nCandidate FAILED its integration tests — live model left untouched."); return 2
    if not BACKUP.exists() and LIVE.exists():                 # rollback copy of whatever is live now
        shutil.copy2(LIVE, BACKUP)
    if not BSCHEMA.exists() and LSCHEMA.exists():
        shutil.copy2(LSCHEMA, BSCHEMA)
    res = json.loads((ART / "comprehensive_results.json").read_text(encoding="utf8"))
    shutil.copy2(CAND, LIVE)
    shutil.copy2(CSCHEMA, LSCHEMA)
    CAND.unlink(); CSCHEMA.unlink()
    print(f"\nPromoted: {res['selection']['chosen']} is now pre_release_movie_classifier.joblib (previous model preserved as {BACKUP.name}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
