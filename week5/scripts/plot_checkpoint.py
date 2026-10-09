"""Audit Treeverse actions and plot the storage tradeoff and Marmousi image."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def report_reflector():
    reference = np.load(ART / "adjoint/image.npy")
    results = []
    for budget in (1,3,5,10):
        directory = ART / f"checkpoint-{budget}"
        image = np.load(directory / "image.npy")
        stats = json.loads((directory / "result.json").read_text())["statistics"]
        relative_error = np.linalg.norm(image-reference)/np.linalg.norm(reference)
        all_counts = np.zeros(3,dtype=int)
        for shot in stats["per_shot"]:
            actions = json.loads((directory / shot["actions_file"]).read_text())
            saved = {0}
            restores = 0
            overruns = 0
            grads = []
            for action in actions:
                kind,step = action["action"],action["step"]
                if kind == "restore": restores += step not in saved
                elif kind == "store": saved.add(step)
                elif kind == "fetch": saved.remove(step)
                elif kind == "grad": grads.append(step)
                overruns += len(saved) > budget+1 or action["saved_states"] != len(saved)
            all_counts += (sum(a != b for a,b in zip(grads,range(239,-1,-1)))
                           + abs(len(grads)-240), restores, overruns)
        print(f"budget={budget}: relative image error={relative_error:.3g}, "
              f"peak states={stats['peak_saved_states']}, forward calls/shot="
              f"{stats['scheduler_forward_calls']//3}; audit "
              f"(grad, restore, budget)={all_counts.tolist()}")
        if relative_error >= 1e-9 or np.any(all_counts):
            raise RuntimeError(f"checkpoint budget {budget} failed image or action audit")
        results.append((budget,stats))
    return results


def plot_actions():
    actions = json.loads((ART / "checkpoint-5/actions-0.json").read_text())
    fig,ax = plt.subplots(figsize=(10,4),constrained_layout=True)
    colors = {"store":"orange","restore":"purple","call":"steelblue",
              "grad":"crimson","fetch":"green"}
    for kind,color in colors.items():
        points = [(i,a["step"]) for i,a in enumerate(actions) if a["action"] == kind]
        ax.scatter(*zip(*points),s=3 if kind == "call" else 10,color=color,label=kind)
    ax.set(xlabel="Operation index",ylabel="Time step",title="Treeverse schedule, reflector shot 0, budget 5")
    ax.legend(ncol=5,markerscale=2)
    fig.savefig(ART / "checkpoint-actions.png",dpi=160)
    plt.close(fig)


def plot_work(results):
    budgets = [budget for budget,_ in results]
    work = [stats["scheduler_forward_calls"]//3 for _,stats in results]
    saved_bytes = [stats["peak_saved_bytes"] for _,stats in results]
    full = json.loads((ART / "adjoint/result.json").read_text())["statistics"]
    fig,axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].semilogy(budgets,work,"o-",label="Treeverse")
    axes[0].axhline(full["scheduler_forward_calls"]//3,color="gray",ls="--",label="full history")
    axes[0].set(xlabel="Additional checkpoint slots",ylabel="Forward steps per shot")
    axes[0].legend()
    axes[1].plot(budgets,saved_bytes,"o-",label="Treeverse")
    axes[1].axhline(full["peak_saved_bytes"],color="gray",ls="--",label="full history")
    axes[1].set(xlabel="Additional checkpoint slots",ylabel="Peak saved-state bytes")
    axes[1].legend()
    fig.savefig(ART / "checkpoint-work.png",dpi=160)
    plt.close(fig)


def plot_marmousi():
    experiment = json.loads((ROOT / "inputs/marmousi.json").read_text())
    background = np.asarray(experiment["background"])
    perturbation = np.asarray(experiment["perturbation"])
    born = np.load(ART / "marmousi-born/born_data.npy")
    image = np.load(ART / "marmousi-image/image.npy")
    stats = json.loads((ART / "marmousi-image/result.json").read_text())["statistics"]
    norm = np.linalg.norm(image)
    print(f"Marmousi image L2={norm:.10g}, peak saved states={stats['peak_saved_states']}, "
          f"bytes={stats['peak_saved_bytes']}")
    if abs(norm/6.7037741e-4-1) >= 1e-4 or stats["peak_saved_bytes"] != 20_788_320:
        raise RuntimeError("Marmousi norm or saved-state storage missed the reference")
    scale = experiment["dx"]*experiment["length_unit_m"]/1000
    width = experiment["nx"]*scale
    depth = experiment["nz"]*scale
    shot_x = np.asarray(experiment["shots"])[:,0]*scale
    index = int(np.argmin(np.abs(shot_x-10)))
    receiver_x = np.asarray(experiment["receivers"])[:,0]*scale
    duration = experiment["steps"]*experiment["dt"]*experiment["time_unit_s"]
    fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    panels = [(background,"Smoothed Marmousi background","viridis",None),
              (perturbation,"Short-wavelength perturbation","seismic",None),
              (born[index],f"Born gather; source x={shot_x[index]:.1f} km","seismic",
               [receiver_x[0],receiver_x[-1],duration,0]),
              (image,"Checkpointed migration image","seismic",None)]
    for ax,(data,title,cmap,special_extent) in zip(axes.flat,panels):
        extent = special_extent or [0,width,depth,0]
        limit = np.max(np.abs(data)) if cmap == "seismic" else None
        picture = ax.imshow(data,origin="upper",aspect="auto",extent=extent,cmap=cmap,
                            vmin=-limit if limit is not None else None,
                            vmax=limit if limit is not None else None)
        ax.set(title=title,xlabel="Receiver position (km)" if special_extent else "Horizontal position (km)",
               ylabel="Time (s)" if special_extent else "Depth (km)")
        if special_extent is None and title != "Checkpointed migration image":
            ax.add_patch(Rectangle((8,0),6,3,fill=False,edgecolor="black",ls="--"))
        fig.colorbar(picture,ax=ax,shrink=.75)
    fig.savefig(ART / "marmousi.png",dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results = report_reflector()
    plot_actions()
    plot_work(results)
    plot_marmousi()
