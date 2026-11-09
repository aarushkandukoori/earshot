"""Exact Kaldi fbank kept in float32, ResNet trunk (rank-4 tensors) in float16, via coremltools op_selector."""
import glob, sys, time, numpy as np, soundfile as sf, torch, coremltools as ct
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import convert_spk_coreml as C
mdir, ls_dir, out_dir = sys.argv[1:4]
files = sorted(glob.glob(f"{ls_dir}/**/*.flac", recursive=True))
wavs = [sf.read(f, dtype="float32")[0][: 5 * 16000] for f in files[:40]]
wavs = [w for w in wavs if len(w) == 80000][:20]
sd = {k: v for k, v in torch.load(f"{mdir}/wespeaker_pt/resnet34_avg_model.pt", map_location="cpu", weights_only=True).items() if not k.startswith("projection")}
net = C.ResNet34(); net.load_state_dict(sd)
ref = C.Embedder(C.KaldiFbank().eval(), net.eval()).eval()
with torch.no_grad():
    pr = [ref(torch.from_numpy(x)[None]).numpy()[0] for x in wavs]
tr = torch.jit.trace(ref, torch.from_numpy(wavs[0])[None])
def fp16_only_rank4(op):
    return all(getattr(o, "rank", 0) == 4 for o in op.outputs)
ml = ct.convert(tr, inputs=[ct.TensorType(name="wav", shape=(1, ct.RangeDim(16000, 256000, default=80000)))], outputs=[ct.TensorType(name="embedding")],
                convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                compute_precision=ct.transform.FP16ComputePrecision(op_selector=fp16_only_rank4))
ml.save(f"{out_dir}/wespeaker_resnet34_mixed.mlpackage")
for cu in ("CPU_ONLY", "CPU_AND_GPU", "ALL", "CPU_AND_NE"):
    m = ct.models.MLModel(f"{out_dir}/wespeaker_resnet34_mixed.mlpackage", compute_units=getattr(ct.ComputeUnit, cu))
    cs, lat = [], []
    for x, p in zip(wavs, pr):
        t = time.perf_counter(); o = m.predict({"wav": x[None]})["embedding"][0]; lat.append(time.perf_counter() - t)
        cs.append(C.cos(o, p))
    print(f"coreml mixed fp32-front/fp16-trunk {cu}: min cos {min(cs):.4f}, mean {np.mean(cs):.4f}, median predict {1000*np.median(lat[1:]):.1f} ms", flush=True)
