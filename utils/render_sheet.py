"""Render one extra sheet (a mesh under tifxyz/<name>/) over its whole grid: uint8 zarr ves_zarrs2/<name>.zarr + masks/extras/<name>.png.
  python3 -m utils.render_sheet <name> B [--reversed] [--force]      (B = the 0846B volume, A = the 0846A one)
--reversed adds <name> to utils/depth_reversed.json first (the mesh normal points away from the scroll axis: test with the sign of the mean dot product of the
mesh normal cross(d/dcol, d/drow) and the direction to the axis: positive = reversed, as for every 0846B sheet so far); the render is then in reversed depth order."""
import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("name")
parser.add_argument("volume", choices=("A", "B"))
parser.add_argument("--reversed", action="store_true")
parser.add_argument("--secret", action="store_true", help="record the reversal in the git-excluded utils/depth_reversed_local.json instead of the tracked list")
parser.add_argument("--force", action="store_true")
parser.add_argument("--workers", type=int, default=12)
args = parser.parse_args()
if args.reversed:
    path = REPO / "utils" / ("depth_reversed_local.json" if args.secret else "depth_reversed.json")
    if not path.is_file():
        path.write_text(json.dumps({"zids": []}))
    data = json.loads(path.read_text(encoding="utf-8"))
    if args.name not in data["zids"]:
        data["zids"] = sorted({*data["zids"], args.name})
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        os.replace(temporary, path)
import assemble_test_segments as t  # noqa: E402
from utils import mix6  # noqa: E402
from utils.assemble_herculex import VOLUMES  # noqa: E402

os.makedirs(t.EXTRAS_MASK_DIR, exist_ok=True)
print(t.render_fragment(args.name, args.name, *VOLUMES[args.volume], args.workers, mix6.zarr_dir().as_posix(), t.SCRIPT_DIR, t.DEFAULT_CHUNK_DEPTH,
                        t.DEFAULT_CHUNK_Y, t.DEFAULT_CHUNK_X, force=args.force, mask_dir=t.EXTRAS_MASK_DIR, output_dtype=t.EXTRAS_DTYPE, drop_cache=True, crop_valid=False))
