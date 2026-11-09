import collections, json
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
files = sorted(f for f in fs.find("datasets/quinnlue/FSD50K-16k/data") if f.endswith(".parquet"))
lic_h = collections.Counter(); lab_h = collections.Counter(); lab_h_ok = collections.Counter()
total = 0.0; ok_total = 0.0; n = 0
WANT = {"Crowd", "Chatter", "Cheering", "Applause", "Laughter", "Cutlery_and_silverware", "Dishes_and_pots_and_pans",
        "Glass", "Chink_and_clink", "Music", "Walk_and_footsteps", "Clapping", "Door", "Drawer_open_or_close",
        "Typing", "Coin_(dropping)", "Water_tap_and_faucet", "Microwave_oven", "Frying_(food)", "Boiling",
        "Bowed_string_instrument", "Guitar", "Piano", "Drum_kit", "Bass_guitar", "Electric_guitar", "Cough", "Sneeze", "Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Conversation", "Human_voice", "Singing", "Rain", "Traffic_noise_and_roadway_noise", "Car_passing_by", "Vehicle", "Bus", "Subway_and_metro_and_underground", "Train"}
for f in files:
    with fs.open(f, "rb") as fh:
        t = pq.ParquetFile(fh).read(columns=["labels", "license", "duration_seconds", "source_partition"])
    for labs, lic, dur in zip(t.column("labels").to_pylist(), t.column("license").to_pylist(), t.column("duration_seconds").to_pylist()):
        d = (dur or 0) / 3600; total += d; n += 1
        key = "CC0" if "zero" in lic or "publicdomain" in lic else ("CC-BY" if "by/" in lic and "nc" not in lic else ("CC-BY-NC" if "nc" in lic else ("Sampling+" if "sampling" in lic else lic)))
        lic_h[key] += d
        commercial_ok = key in ("CC0", "CC-BY")
        if commercial_ok: ok_total += d
        for l in labs:
            if l in WANT:
                lab_h[l] += d
                if commercial_ok: lab_h_ok[l] += d
print(json.dumps({"files": len(files), "clips": n, "hours_total": round(total, 2), "hours_commercial_ok": round(ok_total, 2),
                  "hours_by_license": {k: round(v, 2) for k, v in lic_h.most_common()},
                  "hours_by_label_all": {k: round(v, 2) for k, v in lab_h.most_common()},
                  "hours_by_label_cc0_ccby": {k: round(v, 2) for k, v in lab_h_ok.most_common()}}, indent=1))
