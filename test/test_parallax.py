import math
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from microlux import make_binary_mag_with_parallax
import VBMicrolensing

jax.config.update("jax_enable_x64", True)


# -----------------------------
# Parameters
# -----------------------------
t0 = 2460000.0
tE = 100.0
rho = 1e-3
u0 = 0.1
q = 1e-3
s = 0.9
alpha_deg = 120.0
piEN = 0.1
piEE = 0.1

# Coordinates (same target, two formats)
ra_deg = 269.7595833
dec_deg = -29.0708889
ra_dec_str = "17:59:02.3 -29:04:15.2"

# Time grid
n_times = 1000
times = jnp.linspace(t0 - 5.0 * tE, t0 + 5.0 * tE, n_times)


# -----------------------------
# microlux model (JAX)
# -----------------------------
tref = t0
binary_mag_par = make_binary_mag_with_parallax(ra=ra_deg, dec=dec_deg, tref=tref)
mag_lux = binary_mag_par(t0, u0, tE, rho, q, s, alpha_deg, piEN, piEE, times)


# -----------------------------
# VBMicrolensing model (NumPy)
# -----------------------------
VBM = VBMicrolensing.VBMicrolensing()
VBM.SetObjectCoordinates(ra_dec_str)

alpha_rad = np.deg2rad(alpha_deg)

# VBMicrolensing parameter vector:
# [log(s), log(q), u0, alpha(rad)-pi, log(rho), log(tE), t0-2450000, piEN, piEE]
pr_vbm = [
    math.log(s),
    math.log(q),
    u0,
    alpha_rad - np.pi,
    math.log(rho),
    math.log(tE),
    t0 - 2450000.0,
    piEN,
    piEE,
]

times_np = np.asarray(times - 2450000.0, dtype=float)
mag_vbm = np.asarray(VBM.BinaryLightCurveParallax(pr_vbm, times_np)[0], dtype=float)


# -----------------------------
# Plot 1: light curves
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.plot(np.asarray(times), np.asarray(mag_lux), label="microlux", linewidth=1.8)
ax.plot(np.asarray(times), mag_vbm, label="VBMicrolensing", linewidth=1.2, alpha=0.9)

ax.set_title("Binary microlensing with parallax: magnification vs time")
ax.set_xlabel("Time (JD)")
ax.set_ylabel("Magnification")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")
plt.show()


# -----------------------------
# Plot 2: relative difference
# -----------------------------
mag_lux_np = np.asarray(mag_lux, dtype=float)
denom = np.maximum(np.abs(mag_vbm), 1e-15)
rel_err = np.abs(mag_vbm - mag_lux_np) / denom

fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
ax.plot(np.asarray(times), rel_err, linewidth=1.4)
ax.set_yscale("log")

ax.set_title("Relative difference |VBM - microlux| / |VBM|")
ax.set_xlabel("Time (JD)")
ax.set_ylabel("Relative error")
ax.grid(True, which="both", alpha=0.3)
plt.show()


# -----------------------------
# Jacobian (JAX)
# -----------------------------
def f(theta):
    t0_, u0_, tE_, rho_, q_, s_, alpha_deg_, piEN_, piEE_ = theta
    return binary_mag_par(t0_, u0_, tE_, rho_, q_, s_, alpha_deg_, piEN_, piEE_, times)


theta0 = jnp.array([t0, u0, tE, rho, q, s, alpha_deg, piEN, piEE], dtype=jnp.float64)
J = jax.jacrev(f)(theta0)  # shape: (n_times, 9)

param_names = ["t0", "u0", "tE", "rho", "q", "s", "alpha", "piEN", "piEE"]

fig, axes = plt.subplots(
    1 + len(param_names),
    1,
    sharex=True,
    figsize=(10, 2.0 * (1 + len(param_names))),
    constrained_layout=True,
)

# top: curves
axes[0].plot(np.asarray(times), mag_lux_np, label="microlux", linewidth=1.8)
axes[0].set_ylabel("A")
axes[0].grid(True, which="both", alpha=0.3)

# derivatives
for i, name in enumerate(param_names):
    ax = axes[i + 1]
    ax.plot(np.asarray(times), np.asarray(J[:, i]), linewidth=1.2)
    ax.axhline(0.0, linewidth=0.9, alpha=0.7)
    ax.set_ylabel(f"dA/d{name}")
    ax.grid(True, which="both", alpha=0.25)

axes[-1].set_xlabel("Time (JD)")
plt.show()

