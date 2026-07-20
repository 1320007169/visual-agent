from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
written = skipped = 0
for source in sorted((ROOT / "data/visual_agent_parquet_snapshot/images").glob("*.parquet")):
    for batch in pq.ParquetFile(source).iter_batches(columns=["path", "bytes"]):
        for row in batch.to_pylist():
            target = ROOT / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == len(row["bytes"]):
                skipped += 1
                continue
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(row["bytes"])
            temporary.replace(target)
            written += 1
print({"written": written, "skipped": skipped})
