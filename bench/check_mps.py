"""Check that each model gives the same output and gradients on MPS as on CPU (fp32),
and that autocast fp16 / bf16 runs on MPS. Prints one JSON line per model."""
import copy
import json

import torch

import models as M


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def main():
    torch.manual_seed(0)
    mix = torch.randn(2, 16000) * 0.1
    tgt = torch.randn(2, 16000) * 0.1
    emb = torch.randn(2, M.EMB_DIM)
    for name in M.ALL:
        cpu = M.build(name)
        mps = copy.deepcopy(cpu).to("mps")
        out = {"model": name}
        y_c = cpu(mix, emb)
        loss_c = M.neg_si_sdr(y_c, tgt)
        loss_c.backward()
        y_m = mps(mix.to("mps"), emb.to("mps"))
        loss_m = M.neg_si_sdr(y_m, tgt.to("mps"))
        loss_m.backward()
        out["out_rel_err"] = rel(y_m.cpu(), y_c.detach())
        g_c = torch.cat([p.grad.flatten() for p in cpu.parameters() if p.grad is not None])
        g_m = torch.cat([p.grad.flatten().cpu() for p in mps.parameters() if p.grad is not None])
        out["grad_rel_err"] = rel(g_m, g_c)
        for dt_name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            try:
                mps.zero_grad()
                with torch.autocast("mps", dtype=dt):
                    y = mps(mix.to("mps"), emb.to("mps"))
                loss = M.neg_si_sdr(y, tgt.to("mps"))
                loss.backward()
                g = torch.cat([p.grad.flatten().cpu() for p in mps.parameters() if p.grad is not None])
                out[f"{dt_name}_out_rel_err"] = rel(y.float().detach().cpu(), y_c.detach())
                out[f"{dt_name}_grad_rel_err"] = rel(g, g_c)
            except Exception as e:  # noqa: BLE001
                out[f"{dt_name}_error"] = f"{type(e).__name__}: {e}"[:300]
        print(json.dumps(out))


if __name__ == "__main__":
    main()
