import glob, sys, numpy as np, soundfile as sf, torch, coremltools as ct
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import convert_spk_coreml as C
mdir, ls_dir, out_dir = sys.argv[1:4]
wavs = [sf.read(f, dtype="float32")[0][: 5 * 16000] for f in sorted(glob.glob(f"{ls_dir}/**/*.flac", recursive=True))[:20]]
wavs = [w for w in wavs if len(w) == 80000][:10]
net = C.ResNet34(); net.load_state_dict({k: v for k, v in torch.load(f"{mdir}/wespeaker_pt/resnet34_avg_model.pt", map_location="cpu", weights_only=True).items() if not k.startswith("projection")})
model = C.Embedder(C.KaldiFbank().eval(), net.eval()).eval()
with torch.no_grad():
    pt = [model(torch.from_numpy(x)[None]).numpy()[0] for x in wavs]
for cu in ("CPU_ONLY", "CPU_AND_GPU", "ALL"):
    ml = ct.models.MLModel(f"{out_dir}/wespeaker_resnet34_fp32.mlpackage", compute_units=getattr(ct.ComputeUnit, cu))
    cs = [C.cos(ml.predict({"wav": x[None]})["embedding"][0], p) for x, p in zip(wavs, pt)]
    print(cu, "per-wav cos:", [round(c, 4) for c in cs])
