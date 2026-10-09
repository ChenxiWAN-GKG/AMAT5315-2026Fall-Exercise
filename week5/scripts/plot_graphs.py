"""Draw the operations in JAX's pair-energy and gradient jaxprs."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lennard_jones_ad import pair_energy

ROOT = Path(__file__).resolve().parents[1]


def draw(function, destination, title):
    jaxpr = jax.make_jaxpr(function)(jnp.float64(1.3)).jaxpr
    equations = jaxpr.eqns
    producer = {str(jaxpr.invars[0]): -1}
    levels = {-1: 0}
    for index,equation in enumerate(equations):
        levels[index] = max((levels[producer[str(var)]] for var in equation.invars
                             if str(var) in producer),default=0)+1
        for variable in equation.outvars:
            producer[str(variable)] = index
    groups = {}
    for index,level in levels.items(): groups.setdefault(level,[]).append(index)
    positions = {}
    for level,items in groups.items():
        for row,index in enumerate(items):
            positions[index] = (level*1.8, (len(items)-1)/2-row)
    fig,ax = plt.subplots(figsize=(max(8,1.9*(max(levels.values())+1)),max(4,1.0*max(map(len,groups.values()))+1)),
                          constrained_layout=True)
    for index,equation in enumerate(equations):
        x,y = positions[index]
        label = equation.primitive.name
        if label == "integer_pow": label += f" ({equation.params['y']})"
        for variable in equation.invars:
            earlier = producer.get(str(variable))
            if earlier is None or earlier == index: continue
            xx,yy = positions[earlier]
            ax.annotate("",(x-.35,y),(xx+.35,yy),arrowprops={"arrowstyle":"->","lw":1,"color":"#59636e"})
        ax.text(x,y,label,ha="center",va="center",fontsize=10,
                bbox={"boxstyle":"round,pad=.4","facecolor":"#f6b4a9" if label == "add_any" else "#cfe8fa",
                      "edgecolor":"#35536a"})
    x,y = positions[-1]
    ax.text(x,y,"r",ha="center",va="center",fontsize=11,
            bbox={"boxstyle":"round,pad=.4","facecolor":"#e3edcd","edgecolor":"#47683e"})
    ax.set(xlim=(-.7,1.8*max(levels.values())+.8),ylim=(-max(map(len,groups.values()))/2-1,
        max(map(len,groups.values()))/2+1),title=title)
    ax.axis("off")
    fig.savefig(destination,dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    output = ROOT / "artifacts/ad"
    output.mkdir(parents=True,exist_ok=True)
    draw(pair_energy,output/"graph.png","JAX operations for Lennard-Jones pair energy")
    draw(jax.grad(pair_energy),output/"grad-graph.png","JAX gradient graph: add_any joins both paths from a")
