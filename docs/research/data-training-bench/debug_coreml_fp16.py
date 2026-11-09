"""fp16-safe variant of the in-graph fbank: CMN removes any constant gain, so drop the x32768 scaling,
pre-scale by 1/8 and clamp at 1e-4 to stay inside float16 range. Compare to the exact fp32 PyTorch model."""
import glob, sys, numpy as np, soundfile as sf, torch, torch.nn.functional as F, coremltools as ct
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import convert_spk_coreml as C

class Fp16SafeFbank(C.KaldiFbank):
    def forward(self, wav):
        spec = F.conv1d((wav * 0.125).unsqueeze(1), self.kern, stride=self.hop)
        re, im = spec[:, : self.nb], spec[:, self.nb:]
        fb = torch.log(torch.clamp(torch.matmul(self.mel, re * re + im * im), min=1e-4))
        fb = fb - fb.mean(dim=2, keepdim=True)
        return fb.transpose(1, 2)

mdir, ls_dir = sys.argv[1:3]
files = sorted(glob.glob(f"{ls_dir}/**/*.flac", recursive=True))
wavs = [sf.read(f, dtype="float32")[0][: 5 * 16000] for f in files[:40]]
wavs = [w for w in wavs if len(w) == 80000][:20]
sd = {k: v for k, v in torch.load(f"{mdir}/wespeaker_pt/resnet34_avg_model.pt", map_location="cpu", weights_only=True).items() if not k.startswith("projection")}
net = C.ResNet34(); net.load_state_dict(sd)
ref = C.Embedder(C.KaldiFbank().eval(), net.eval()).eval()
safe = C.Embedder(Fp16SafeFbank().eval(), net.eval()).eval()
with torch.no_grad():
    pr = [ref(torch.from_numpy(x)[None]).numpy()[0] for x in wavs]
    ps = [safe(torch.from_numpy(x)[None]).numpy()[0] for x in wavs]
print("pytorch fp32 safe-front vs exact-front, min cos", round(min(C.cos(a, b) for a, b in zip(ps, pr)), 5))
tr = torch.jit.trace(safe, torch.from_numpy(wavs[0])[None])
ml = ct.convert(tr, inputs=[ct.TensorType(name="wav", shape=(1, ct.RangeDim(16000, 256000, default=80000)))], outputs=[ct.TensorType(name="embedding")],
                convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16)
for cu in ("CPU_ONLY", "CPU_AND_GPU", "ALL", "CPU_AND_NE"):
    m = ct.models.MLModel(ml.get_spec(), weights_dir=ml.weights_dir, compute_units=getattr(ct.ComputeUnit, cu))
    import time
    cs, lat = [], []
    for x, p in zip(wavs, pr):
        t = time.perf_counter(); o = m.predict({"wav": x[None]})["embedding"][0]; lat.append(time.perf_counter() - t)
        cs.append(C.cos(o, p))
    print(f"coreml fp16 {cu}: min cos vs exact fp32 {min(cs):.4f}, mean {np.mean(cs):.4f}, median predict {1000*np.median(lat[1:]):.1f} ms", flush=True)
