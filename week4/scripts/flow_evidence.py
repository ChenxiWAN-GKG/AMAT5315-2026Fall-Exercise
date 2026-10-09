"""Regenerate the Week 4 flow runs and their evidence plots."""

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release"
ART = ROOT / "artifacts"
EVIDENCE = ROOT / "evidence"


def field(case, n, **flags):
    command = [str(BIN / "field"), case, "--n", str(n)]
    for key, value in flags.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def run(name, initial, method, nu, dt, end, every):
    destination = ART / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [str(BIN / "fluid"), "--method", method, "--nu", str(nu),
               "--dt", str(dt), "--t-end", str(end), "--every", str(every),
               "--out", str(destination)]
    result = subprocess.run(command, input=json.dumps(initial), capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    (ART / (name + ".tsv")).write_text(result.stdout)
    rows = np.genfromtxt(ART / (name + ".tsv"), delimiter="\t", names=True)
    if rows.ndim == 0:
        rows = np.array([rows], dtype=rows.dtype)
    return rows, result.returncode


def frame(name, time=None):
    path = ART / name / "fields.jsonl"
    selected = None
    with path.open() as source:
        for line in source:
            item = json.loads(line)
            if time is None or abs(item["t"] - time) < 1e-8:
                selected = item
                if time is not None:
                    break
    if selected is None:
        raise ValueError(f"no frame at t={time} in {name}")
    for key in ("u", "v", "omega"):
        selected[key] = np.asarray(selected[key], dtype=float)
    return selected


def relative(first, second, key):
    a = first[key]
    b = second[key]
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def perturb(initial):
    n = initial["n"]
    u = np.asarray(initial["u"]).reshape(n, n)
    v = np.asarray(initial["v"]).reshape(n, n)
    scale = max(np.abs(u).max(), np.abs(v).max())
    x = 2*np.pi*np.arange(n)/n
    ripple = -7e-5*scale*np.cos(4*x[:, None])*np.cos(3*x[None, :])
    k = np.fft.fftfreq(n, d=1/n)
    kx, ky = np.meshgrid(k, k)
    k2 = kx*kx+ky*ky
    k2[0, 0] = 1
    modes = np.fft.fft2(ripple)
    du = np.fft.ifft2(1j*ky/k2*modes).real
    dv = np.fft.ifft2(-1j*kx/k2*modes).real
    modified = dict(initial)
    modified["u"] = (u+du).ravel().tolist()
    modified["v"] = (v+dv).ravel().tolist()
    return modified


def plot_fields():
    t0, t1 = frame("taylor-green", 0), frame("taylor-green", 1)
    exact = json.loads((ART / "taylor-green" / "exact-t1.json").read_text())
    calculated = np.r_[t1["u"], t1["v"]]
    expected = np.r_[exact["u"], exact["v"]]
    error = np.linalg.norm(calculated-expected)/np.linalg.norm(expected)
    print(f"Taylor-Green velocity relative error: {error:.8g}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, item, label in zip(axes, (t0, t1), ("t = 0", "t = 1")):
        n = int(np.sqrt(item["omega"].size))
        image = ax.imshow(item["omega"].reshape(n,n), origin="lower", extent=(0,2*np.pi,0,2*np.pi),
                          cmap="coolwarm", vmin=-2, vmax=2)
        grid = np.arange(0,n,8)
        xx, yy = np.meshgrid(2*np.pi*grid/n,2*np.pi*grid/n)
        ax.quiver(xx,yy,item["u"].reshape(n,n)[::8,::8],item["v"].reshape(n,n)[::8,::8], color="black")
        ax.set(title=label, xlabel="x", ylabel="y", xticks=[0,np.pi,2*np.pi], yticks=[0,np.pi,2*np.pi],
               xticklabels=["0","π","2π"], yticklabels=["0","π","2π"])
    fig.colorbar(image, ax=axes, label="vorticity ω")
    fig.savefig(EVIDENCE / "taylor-green.png", dpi=160)
    plt.close(fig)

    records = [frame("random", t) for t in (0,2,5,10)]
    limit = max(np.max(np.abs(item["omega"])) for item in records)
    fig, axes = plt.subplots(1,4,figsize=(14,3.5),constrained_layout=True)
    for ax,item,t in zip(axes,records,(0,2,5,10)):
        n = int(np.sqrt(item["omega"].size))
        image = ax.imshow(item["omega"].reshape(n,n),origin="lower",cmap="coolwarm",vmin=-limit,vmax=limit)
        ax.set(title=f"t = {t}",xlabel="x",ylabel="y")
    fig.colorbar(image,ax=axes,label="vorticity ω")
    fig.savefig(EVIDENCE / "random.png",dpi=160)
    plt.close(fig)


def plot_blowup(tg, random):
    scan = {}
    for label, initial, nu, end, cases in (
        ("taylor-green",tg,0.1,8,[("rk4",0.032),("rk4",0.033)]),
        ("random",random,0.004,10,[("rk4",0.038),("rk4",0.040),("euler",0.01)]),
    ):
        for method,dt in cases:
            name = f"scan/{label}-{method}-{dt}"
            scan[(label,method,dt)] = run(name,initial,method,nu,dt,end,0.5)
    safe = [dt for dt in (0.038,0.040) if scan[("random","rk4",dt)][1] == 0]
    unstable = [dt for dt in (0.038,0.040) if scan[("random","rk4",dt)][1] == 1]
    for dt in (0.035,0.03,0.045,0.05,0.06):
        if safe and unstable: break
        key = ("random","rk4",dt)
        scan[key] = run(f"scan/random-rk4-{dt}",random,"rk4",0.004,dt,10,0.5)
        (safe if scan[key][1] == 0 else unstable).append(dt)
    if not safe or not unstable:
        raise RuntimeError("random RK4 stability bracket not found")
    chosen = [("taylor-green","rk4",0.032),("taylor-green","rk4",0.033),
              ("random","rk4",max(safe)),("random","rk4",min(unstable)),("random","euler",0.01)]
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,label in zip(axes,("taylor-green","random")):
        for case,method,dt in chosen:
            if case != label: continue
            rows,exit_code = scan[(case,method,dt)]
            finite = np.isfinite(rows["E"]) & (rows["E"] > 0)
            ax.plot(rows["t"][finite],rows["E"][finite],label=f"{method}, Δt={dt}")
            if exit_code:
                ax.axvline(rows["t"][-1],color="gray",ls=":",alpha=.5)
                ax.scatter(rows["t"][finite][-1],rows["E"][finite][-1],marker="x")
            print(f"{case} {method} dt={dt}: exit={exit_code}, last t={rows['t'][-1]:.3g}")
        ax.set(title=label,xlabel="time t",ylabel="energy E",yscale="log")
        ax.legend(fontsize=8)
    fig.savefig(EVIDENCE / "blowup.png",dpi=160)
    plt.close(fig)


def plot_sensitivity(tg, random):
    fig, ax = plt.subplots(figsize=(7,4),constrained_layout=True)
    for label,initial,nu in (("taylor-green",tg,0.1),("random",random,0.004)):
        a = f"sensitivity/{label}-base"
        b = f"sensitivity/{label}-ripple"
        run(a,initial,"rk4",nu,0.01,20,0.5)
        run(b,perturb(initial),"rk4",nu,0.01,20,0.5)
        with (ART/a/"fields.jsonl").open() as left, (ART/b/"fields.jsonl").open() as right:
            pairs = [(json.loads(x),json.loads(y)) for x,y in zip(left,right)]
        times = [x["t"] for x,y in pairs]
        errors = [np.linalg.norm(np.asarray(x["omega"])-y["omega"])/np.linalg.norm(x["omega"])
                  for x,y in pairs]
        ax.plot(times,errors,label=label)
        print(f"{label} sensitivity: {errors[0]:.6g} -> {errors[-1]:.6g}")
    ax.set(xlabel="time t",ylabel="relative vorticity separation",yscale="log")
    ax.legend()
    fig.savefig(EVIDENCE/"sensitivity.png",dpi=160)
    plt.close(fig)


def plot_order():
    initial = field("taylor-green",8)
    exact = field("taylor-green",8,nu=0.5,t=2)
    expected = np.r_[exact["u"],exact["v"]]
    dts = np.array([0.4,0.25,0.2])
    errors = []
    for dt in dts:
        name = f"order/rk4-dt{dt}"
        run(name,initial,"rk4",0.5,dt,2,2)
        last = frame(name)
        errors.append(np.linalg.norm(np.r_[last["u"],last["v"]]-expected)/np.linalg.norm(expected))
    slope,offset = np.polyfit(np.log(dts),np.log(errors),1)
    print(f"Taylor-Green RK4 order slope: {slope:.4f}; errors: {errors}")
    fig,ax = plt.subplots(figsize=(6,4),constrained_layout=True)
    ax.loglog(dts,errors,"o",label="computed")
    ax.loglog(dts,np.exp(offset)*dts**slope,"--",label=f"fit, slope {slope:.3f}")
    ax.axhline(1e-6,color="gray",ls=":",label="six-decimal storage scale")
    ax.set(xlabel="time step Δt",ylabel="relative velocity error at t=2")
    ax.legend()
    fig.savefig(EVIDENCE/"order.png",dpi=160)
    plt.close(fig)
    return slope


def plot_convergence(random):
    dts = np.array([0.02,0.0125,0.01])
    recorded = {}
    for dt in [*dts,0.0025]:
        name = f"convergence/rk4-dt{dt}"
        run(name,random,"rk4",0.004,dt,2,2)
        recorded[dt] = frame(name)
    reference = recorded[0.0025]
    errors = np.array([relative(recorded[dt],reference,"omega") for dt in dts])
    slope,offset = np.polyfit(np.log(dts),np.log(errors),1)
    estimate = relative(recorded[0.02],recorded[0.01],"omega")/15
    predicted = np.array([estimate*(dt/0.01)**4 for dt in dts])
    eligible = [dt for dt,prediction in zip(dts,predicted) if prediction < 5e-6]
    chosen = max(eligible) if eligible else None
    data = {"reference_dt":0.0025,"time":2,"n":128,"dt":dts.tolist(),
            "relative_omega_error":errors.tolist(),"slope":float(slope),
            "richardson_at_0.01":estimate,"predicted_error":predicted.tolist(),
            "threshold":5e-6,"chosen_dt":chosen}
    (EVIDENCE/"convergence.json").write_text(json.dumps(data,indent=2)+"\n")
    print(f"random RK4 slope: {slope:.4f}; chosen dt: {chosen}; predicted: {predicted}; measured: {errors}")
    fig,ax = plt.subplots(figsize=(6,4),constrained_layout=True)
    ax.loglog(dts,errors,"o",label=f"measured, slope {slope:.3f}")
    ax.loglog(dts,np.exp(offset)*dts**slope,"--",label="log-log fit")
    ax.loglog(dts,predicted,"x:",label="Richardson prediction")
    ax.axhline(5e-6,color="gray",ls="--",label="target 5×10⁻⁶")
    if chosen is not None: ax.axvline(chosen,color="green",ls=":",label=f"choice Δt={chosen}")
    ax.set(xlabel="time step Δt",ylabel="relative vorticity error at t=2")
    ax.legend()
    fig.savefig(EVIDENCE/"convergence.png",dpi=160)
    plt.close(fig)
    return slope


def main():
    EVIDENCE.mkdir(exist_ok=True)
    tg = field("taylor-green",64)
    random = field("random",128,seed=2026,k_min=2,k_max=6)
    speed = np.sqrt(np.asarray(random["u"])**2+np.asarray(random["v"])**2).max()
    print(f"random initial largest speed: {speed:.6f}")
    plot_fields()
    plot_blowup(tg,random)
    plot_sensitivity(tg,random)
    plot_order()
    plot_convergence(random)


if __name__ == "__main__":
    main()
