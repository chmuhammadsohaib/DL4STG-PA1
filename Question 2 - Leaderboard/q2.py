import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from autoformer import Autoformer
torch.set_num_threads(1)
root = Path(__file__).resolve().parent
res = root / "results"
(res / "models").mkdir(parents=True, exist_ok=True)
y = pd.read_csv(root / "Data/student_train.csv")["value"].to_numpy(float)
ext = pd.read_csv(root / "Data/optional_external_data.csv")
n, hz, trainEnd, valEnd = len(y), 168, 34920, 39288
valEnds = np.arange(trainEnd, valEnd - hz + 1, 24)
mu, sd, floor = y[:trainEnd].mean(), y[:trainEnd].std(), np.quantile(y[:trainEnd], 0.05)
z = torch.tensor((y - mu) / sd, dtype=torch.float32)
cov = ext.drop(columns="time_idx").to_numpy(float)
cov[:, 3:6] = np.log1p(cov[:, 3:6])
cov[:, :6] = (cov[:, :6] - cov[:trainEnd, :6].mean(0)) / cov[:trainEnd, :6].std(0)
tt = np.arange(len(ext))
tf = torch.tensor(np.stack([np.sin(2 * np.pi * tt / 24), np.cos(2 * np.pi * tt / 24), np.sin(4 * np.pi * tt / 24), np.cos(4 * np.pi * tt / 24)], 1), dtype=torch.float32)
full = torch.cat([tf, torch.tensor(cov, dtype=torch.float32)], 1)
marks = {"none": (tf, tf), "past": (full, tf), "future": (full, full)}
def batch(name, ends):
    e = torch.as_tensor(np.asarray(ends))[:, None]
    iE, iD = e + torch.arange(-168, 0), e + torch.arange(-84, hz)
    return z[iE].unsqueeze(-1), marks[name][0][iE], marks[name][1][iD], (z[e + torch.arange(hz)] if np.max(ends) + hz <= n else None)
def predict(model, x, mE, mD):
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(x[i:i + 256], mE[i:i + 256], mD[i:i + 256]) for i in range(0, len(x), 256)]).numpy() * sd + mu
    return np.clip(out, 0, None)
def score(pred, ends):
    tgt = y[np.asarray(ends)[:, None] + np.arange(hz)]
    err, den = pred - tgt, np.abs(tgt) + np.abs(pred)
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean(1)).mean()), smape=float(100 * np.mean(np.divide(2 * np.abs(err), den, out=np.zeros_like(den), where=den > 0))))
def fit(name, seed, fitEnd, epochs, stopEnds=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Autoformer(168, 84, hz, marks[name][0].shape[1], marks[name][1].shape[1])
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    ends = np.arange(168, fitEnd - hz + 1)
    best, state, bestEp, wait, t0 = np.inf, None, 0, 0, time.time()
    for ep in range(1, epochs + 1):
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 ** (ep - 1)
        model.train()
        tot = 0.0
        for idx in np.array_split(rng.permutation(ends), int(np.ceil(len(ends) / 64))):
            x, mE, mD, tgt = batch(name, idx)
            loss = ((model(x, mE, mD) - tgt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        if stopEnds is None:
            print(f"epoch {ep}: train loss {tot / len(ends):.4f}", flush=True)
            continue
        r = score(predict(model, *batch(name, stopEnds)[:3]), stopEnds)["rmse"]
        print(f"epoch {ep}: train loss {tot / len(ends):.4f}, validation RMSE {r:.2f}", flush=True)
        if r < best:
            best, state, bestEp, wait = r, {key: v.clone() for key, v in model.state_dict().items()}, ep, 0
        else:
            wait += 1
            if wait >= 2:
                break
    if state is not None:
        model.load_state_dict(state)
    return model, dict(bestEpoch=bestEp or ep, epochsRun=ep, seconds=time.time() - t0, params=sum(q.numel() for q in model.parameters() if q.requires_grad))
mode = sys.argv[1]
if mode == "eda":
    yc = y - y.mean()
    acf = np.array([1.0] + [np.dot(yc[:-k], yc[k:]) / np.dot(yc, yc) for k in range(1, 401)])
    spec, fr = np.abs(np.fft.rfft(yc, 2 ** 20)) ** 2, np.fft.rfftfreq(2 ** 20)
    day, slow = (fr > 1 / 30) & (fr < 1 / 18), (fr > 1 / 12000) & (fr < 1 / 6000)
    dayLen, slowLen = 1 / fr[day][spec[day].argmax()], 1 / fr[slow][spec[slow].argmax()]
    pw = np.abs(np.fft.rfft(yc)) ** 2
    per = n / np.arange(1, len(pw))
    edges = np.logspace(np.log10(4), np.log10(45000), 121)
    idx = np.digitize(per, edges)
    env = np.array([pw[1:][idx == i].max() if (idx == i).any() else np.nan for i in range(1, len(edges))]) / np.median(pw[1:])
    mid = np.sqrt(edges[1:] * edges[:-1])
    ok = ~np.isnan(env)
    mid, env = mid[ok], env[ok]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax[0].plot(acf)
    for lag in (24, 48, 168):
        ax[0].annotate(f"lag {lag}: {acf[lag]:.2f}", (lag, acf[lag]), (lag + 12, acf[lag] + 0.18), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set(xlabel="lag (steps)", ylabel="autocorrelation", title="(a) Autocorrelation")
    ax[1].loglog(mid, env, lw=1.2)
    for pk, off in ((24, (8, 4)), (12, (-14, 6))):
        j = np.argmin(np.abs(np.log(mid) - np.log(pk)))
        ax[1].annotate(str(pk), (mid[j], env[j]), off, textcoords="offset points", fontsize=8)
    j = np.argmin(np.abs(np.log(mid) - np.log(slowLen)))
    ax[1].annotate(f"broad peak ~{slowLen:,.0f}", (mid[j], env[j]), (0.98, 0.12), textcoords="axes fraction", ha="right", fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax[1].set(xlabel="period (steps)", ylabel="max power / median", title="(b) Periodogram, log-binned maxima")
    fig.tight_layout()
    (root / "figures").mkdir(exist_ok=True)
    fig.savefig(root / "figures" / "t2_periodicity.pdf", bbox_inches="tight")
    print(dict(dayLen=float(dayLen), slowLen=float(slowLen), acf={k: round(float(acf[k]), 3) for k in (1, 12, 24, 48, 168)}))
if mode == "tau":
    for tau in (6, 12, 24, 48):
        print(tau, {k: round(v, 2) for k, v in score(mu + (y[valEnds - 1][:, None] - mu) * np.exp(-np.arange(1, hz + 1) / tau), valEnds).items()})
if mode == "exp":
    for seed in map(int, sys.argv[3:]):
        model, info = fit(sys.argv[2], seed, trainEnd, 6, valEnds)
        p = predict(model, *batch(sys.argv[2], valEnds)[:3])
        row = dict(marks=sys.argv[2], seed=seed, **info, **score(p, valEnds), **{k + "Floor": v for k, v in score(np.clip(p, floor, None), valEnds).items()})
        pd.DataFrame([row]).to_csv(res / "runs.csv", mode="a", header=not (res / "runs.csv").exists(), index=False)
        torch.save(model.state_dict(), res / "models" / f"{sys.argv[2]}-{seed}.pt")
        print(row, flush=True)
if mode == "importance":
    x, mE, mD, _ = batch("future", valEnds)
    rows = []
    for seed in map(int, sys.argv[2:]):
        model = Autoformer(168, 84, hz, 14, 14)
        model.load_state_dict(torch.load(res / "models" / f"future-{seed}.pt"))
        ref = score(predict(model, x, mE, mD), valEnds)["rmse"]
        for rep in range(3):
            rng = np.random.default_rng(1000 * seed + rep)
            for g, cols in {"A": [4], "B": [5], "C": [6], "D": [7], "E": [8], "F": [9], "G-J": [10, 11, 12, 13]}.items():
                pm = torch.as_tensor(rng.permutation(len(valEnds)))
                mE2, mD2 = mE.clone(), mD.clone()
                mE2[:, :, cols], mD2[:, :, cols] = mE[pm][:, :, cols], mD[pm][:, :, cols]
                rows.append(dict(seed=seed, rep=rep, group=g, increase=score(predict(model, x, mE2, mD2), valEnds)["rmse"] - ref))
    out = pd.DataFrame(rows)
    out.to_csv(res / "importance.csv", index=False)
    print(out.groupby("group")["increase"].agg(["mean", "std"]).round(2).to_string())
if mode == "final":
    seed = int(sys.argv[2])
    run = pd.read_csv(res / "runs.csv").query(f"marks == 'future' and seed == {seed}").iloc[0]
    model, info = fit("future", seed, n, int(run.bestEpoch))
    fc = np.clip(predict(model, *batch("future", [n])[:3])[0], floor, None)
    (res / "forecast.txt").write_text(", ".join(f"{v:.2f}" for v in fc))
    torch.save(model.state_dict(), res / "models" / f"final-{seed}.pt")
    out = dict(seed=seed, refitEpochs=int(run.bestEpoch), floor=float(floor), P=info["params"], E=int(run.epochsRun + run.bestEpoch))
    json.dump(out, open(res / "final.json", "w"), indent=1)
    print(out)
