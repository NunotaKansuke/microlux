import numpy as np
import matplotlib.pyplot as plt
import math
import jax
import jax.numpy as jnp
from microlux import make_binary_mag_with_parallax

import VBMicrolensing
VBM = VBMicrolensing.VBMicrolensing()
jax.config.update("jax_enable_x64", True)

t0 = 2460000
tE = 100
rho = 0.001
u0 = 0.1
q = 0.001
s= 0.9
alpha_deg = 120
alpha_rad = np.deg2rad(alpha_deg)
piEN = 0.1
piEE = 0.1

VBM.SetObjectCoordinates("17:59:02.3 -29:04:15.2")
pr_vbm = [math.log(s), math.log(q), u0, alpha_rad-np.pi, math.log(rho), math.log(tE), t0-2450000, piEN, piEE]
mag_vbm = np.array(VBM.BinaryLightCurveParallax(pr_vbm,times-2450000)[0])

ra = 269.7595833    
dec = -29.0708889
tref = t0
binary_mag_par = make_binary_mag_with_parallax(ra=ra, dec=dec, tref=tref)
mag_lux = binary_mag_par(
    t_0=t0,
    u_0=u0,
    t_E=tE,
    rho=rho,
    q=q,
    s=s,
    alpha_deg=alpha_deg,
    times=times,
    piEN=piEN,
    piEE=piEE,
)
