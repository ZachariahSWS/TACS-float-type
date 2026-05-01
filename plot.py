import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

path = "./py2d_output/output.h5"
out_path = "omega_animation.gif"

frame_stride = 5
fps = 20

with h5py.File(path, "r") as f:
    omega = f["fields/omega"][::frame_stride]

vmin = np.percentile(omega, 1)
vmax = np.percentile(omega, 99)

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(omega[0], origin="lower", vmin=vmin, vmax=vmax)
ax.set_xlabel("x")
ax.set_ylabel("y")
title = ax.set_title("frame 0")
fig.colorbar(im, ax=ax)

def update(i):
    im.set_data(omega[i])
    title.set_text(f"frame {i * frame_stride}")
    return [im, title]

ani = FuncAnimation(fig, update, frames= 1000, interval=1000 / fps, blit=False)
ani.save(out_path, writer=PillowWriter(fps=fps))
print(f"saved {out_path}")
print(f"fixed color scale: vmin={vmin:.6g}, vmax={vmax:.6g}")
