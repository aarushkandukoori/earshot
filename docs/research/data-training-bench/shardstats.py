"""Read only small metadata columns of remote HF parquet shards (HTTP range reads) to count speakers per shard."""
import sys, json, collections
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
def stats(path, cols):
    with fs.open(path, "rb", block_size=1 << 20) as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        names = pf.schema_arrow.names
        use = [c for c in cols if c in names]
        t = pf.read(columns=use)
    d = {"path": path, "rows": md.num_rows, "row_groups": md.num_row_groups, "columns": names}
    for c in use:
        col = t.column(c).to_pylist()
        if c in ("speaker_id", "speaker", "accent", "predicted_accent", "common_voice_accents", "gender"):
            cnt = collections.Counter(col)
            d[c + "_unique"] = len(cnt)
            if "accent" in c:
                d[c + "_top"] = cnt.most_common(12)
        if c in ("duration", "audio_duration"):
            d["hours"] = round(sum(x or 0 for x in col) / 3600, 2)
    return d
for spec in sys.argv[1:]:
    path, _, cols = spec.partition("@")
    try:
        print(json.dumps(stats(path, cols.split(",")), default=str)[:1500], flush=True)
    except Exception as e:
        print(path, "ERR", type(e).__name__, str(e)[:300], flush=True)
