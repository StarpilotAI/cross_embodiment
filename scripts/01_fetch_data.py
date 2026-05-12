"""Pull the YAMGripper_MatFolding episode from HF, extract scene_rgb_2 frames + state parquet."""
from pathlib import Path
from huggingface_hub import snapshot_download
import subprocess
import pyarrow.parquet as pq
import json

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

print("[1/3] snapshot_download...")
repo_dir = snapshot_download(
    repo_id="starpilot-ai/YAMGripper_MatFolding",
    repo_type="dataset",
    local_dir=str(DATA / "hf"),
)
print(f"  -> {repo_dir}")

print("[2/3] locate scene_rgb_2 mp4 + parquet ...")
hf = Path(repo_dir)
videos = list(hf.rglob("observation.images.scene_rgb_2/**/*.mp4"))
parquets = list(hf.rglob("data/**/*.parquet"))
infos = list(hf.rglob("meta/info.json"))
print(f"  scene_rgb_2 mp4s: {videos}")
print(f"  parquets: {parquets}")
print(f"  infos: {infos}")

assert videos and parquets, "expected one mp4 + one parquet"
mp4 = videos[0]
pq_path = parquets[0]

print("[3/3] decoding frames ...")
frames_dir = DATA / "frames_scene_rgb_2"
frames_dir.mkdir(parents=True, exist_ok=True)
# Already-decoded?
existing = sorted(frames_dir.glob("*.png"))
if len(existing) < 10:
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp4),
        "-start_number", "0",
        str(frames_dir / "%06d.png")
    ], check=True)
print(f"  frames written to {frames_dir} ({len(list(frames_dir.glob('*.png')))} files)")

# Read parquet, save lightweight JSON of state/action per frame
print("  parquet -> json ...")
tbl = pq.read_table(pq_path)
df = tbl.to_pandas()
print(f"  parquet rows: {len(df)}  cols: {list(df.columns)}")
# Save the per-frame state + action as a list
records = []
for i, row in df.iterrows():
    rec = {
        "frame": int(row.get("frame_index", i)),
        "timestamp": float(row["timestamp"]) if "timestamp" in df.columns else None,
        "action": row["action"].tolist() if "action" in df.columns else None,
        "state": row["observation.state"].tolist() if "observation.state" in df.columns else None,
    }
    records.append(rec)
(DATA / "frames_meta.json").write_text(json.dumps(records))
print(f"  state/action written for {len(records)} frames")

# Also dump dataset info
if infos:
    info = json.loads(infos[0].read_text())
    (DATA / "info.json").write_text(json.dumps(info, indent=2))
    print(f"  info.json saved; features: {list(info.get('features', {}).keys())}")
print("done")
