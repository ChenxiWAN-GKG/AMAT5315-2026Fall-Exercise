"""Check reflector wavefields and plot the adjoint image."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def frame_peak(name, step):
    folder = ART / "forward"
    recording = json.loads((folder / "run.json").read_text())["recording"]
    index = recording["steps"].index(step)
    data = np.load(folder / f"{name}.npy")[index]
    return float(np.max(np.abs(data)))


def plot_image():
    experiment = json.loads((ROOT / "inputs/reflector.json").read_text())
    perturbation = np.asarray(experiment["perturbation"])
    image = np.load(ART / "adjoint/image.npy")
    born = np.load(ART / "born/born_data.npy")
    left = np.sum(born*born)
    right = np.sum(perturbation*image)
    print(f"transpose identity: {left:.10g}, {right:.10g}; relative difference="
          f"{abs(left-right)/max(abs(left),abs(right)):.3g}")
    if abs(left-right)/max(abs(left),abs(right)) >= 1e-9:
        raise RuntimeError("transpose identity failed")
    window = (slice(10,34),slice(7,34))
    small_perturbation = perturbation[window]
    small_image = image[window]
    profile = np.linalg.norm(small_image,axis=1)
    peak = np.argmax(profile)+10
    print(f"reflector peak row={peak}; true row=21; depth error={abs(peak-21)*0.1:.3g} km")
    if abs(peak-21) > 1: raise RuntimeError("image missed the reflector")
    scale = experiment["dx"]*experiment["length_unit_m"]/1000
    extent=(7*scale,34*scale,34*scale,10*scale)
    fig,axes = plt.subplots(1,3,figsize=(12,5),constrained_layout=True)
    for ax,data,title in zip(axes[:2],(small_perturbation,small_image),
                             ("Known reflector","Raw signed RTM image")):
        limit = np.max(np.abs(data))
        graphic = ax.imshow(data,origin="upper",extent=extent,aspect="auto",cmap="seismic",vmin=-limit,vmax=limit)
        ax.axhline(2.1,color="black",ls="--",lw=.7)
        ax.set(title=title,xlabel="Horizontal position (km)",ylabel="Depth (km)")
        fig.colorbar(graphic,ax=ax,shrink=.75)
    axes[2].plot(profile,np.arange(10,34)*scale)
    axes[2].axhline(peak*scale,color="crimson",ls="--",label=f"peak {peak*scale:.1f} km")
    axes[2].axhline(2.1,color="black",ls=":",label="true 2.1 km")
    axes[2].set(title="Image depth profile",xlabel="Row L2 norm",ylabel="Depth (km)",ylim=(3.4,1.0))
    axes[2].legend()
    fig.savefig(ART / "adjoint/image.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    direct = frame_peak("wavefield",150)
    echo = frame_peak("echo",150)
    print(f"step-150 peak echo/direct = {echo/direct:.4f} ({echo:.6g}/{direct:.6g})")
    plot_image()
