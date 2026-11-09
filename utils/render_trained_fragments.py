"""render every fragment the cue_dropout model trained on (the 12 cue_dropout PHerc0846A sheets + the 14 donor patches) with
0846A_eval.ipynb's setup cells and its render(): prediction | composite, and the inklabel overlay and asf cue.

VESUVIUS_CHECKPOINT=<cue_dropout checkpoint> VESUVIUS_OUT_DIR=<folder> python3 utils/render_trained_fragments.py
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
for path in (REPO, REPO / "old"):
    sys.path.insert(0, str(path))
# utils/platform.py would shadow the stdlib module when this file is run as a script
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]

cells = [cell for cell in json.loads((REPO / "0846A_eval.ipynb").read_text())["cells"] if cell["cell_type"] == "code"]
setup = []
for cell in cells:
    source = "".join(cell["source"])
    if source.startswith("# HELD OUT PRIZE SHEET"):
        break
    setup.append(source)
namespace = {"__name__": "eval_0846A"}
for index, source in enumerate(setup):
    exec(compile(source, f"0846A_eval cell {index}", "exec"), namespace)

mix6 = namespace["mix6"]
sheets = [mix6.SHEETS_0846A[int(alias)][1] for alias in mix6.CUE_TRAIN_0846A]
donors = list(namespace["DONORS"])
names = [*sheets, *donors]
print(f"{len(sheets)} 0846A sheets + {len(donors)} donor patches", flush=True)
namespace["OUT_DIR"].mkdir(parents=True, exist_ok=True)
failed = []
for index, name in enumerate(names, start=1):
    print(f"[{index}/{len(names)}] {name}", flush=True)
    try:
        namespace["render"](name, preview=False)
    except Exception as error:
        failed.append(name)
        print(f"FAILED {name}: {error}", flush=True)
print(f"[done] {len(names) - len(failed)}/{len(names)} rendered into {namespace['OUT_DIR']}", flush=True)
sys.exit(1 if failed else 0)
