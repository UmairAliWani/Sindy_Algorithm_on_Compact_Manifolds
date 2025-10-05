# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
#Import libraries
import warnings
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from scipy.io import loadmat

import pysindy as ps
from pysindy.feature_library import PolynomialLibrary

data = (Path() / "../data").resolve()


@contextmanager
def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    warnings.filters = filters


if __name__ == "testing":
    import sys
    import os

    sys.stdout = open(os.devnull, "w")

# %%
# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

# %%

# %%
dt = 0.02
r = 2
n = r + 1
data_set1=loadmat("PODcoefficients_run1.mat")
data_set2=loadmat("PODcoefficients_run2.mat")
x_train1=np.concatenate((data_set1['alpha'][:5000,:r],data_set1['alphaS'][:5000,0:1]), axis=1 )
x_train2=np.concatenate((data_set2['alpha'][:4000,:r],data_set2['alphaS'][:4000,0:1]), axis=1 )

# %%
t_1=np.arange(0,dt*x_train1.shape[0],dt)
t_2=np.arange(0,dt*x_train2.shape[0],dt)
x_train=[x_train1,x_train2]
t_train=[t_1,t_2]

# %%
optimizer = ps.STLSQ(threshold=1e-4)
library = ps.PolynomialLibrary(degree=5)
model = ps.SINDy(
    optimizer=optimizer,
    feature_library=library,
    feature_names=["x", "y", "z"]
)
model.fit(x_train, t_train, multiple_trajectories=True)
model.print()

# %%
# Simulate the model

x_simulate_run1 = model.simulate(x_train1[0], np.arange(0, 100, 0.02))
x_simulate_run2 = model.simulate(x_train2[0], np.arange(0, 95, 0.02))


# %%
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(121, projection="3d")
ax.plot(x_train1[:, 0], x_train1[:, 1], x_train1[:, 2])
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.title("Full Simulation")

ax = fig.add_subplot(122, projection="3d")
ax.plot(x_simulate_run1[:, 0], x_simulate_run1[:, 1], x_simulate_run1[:, 2])
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.title("Identified System")

# %%
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(121, projection="3d")
ax.plot(x_train2[:, 0], x_train2[:, 1], x_train2[:, 2])
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.title("Full Simulation")

ax = fig.add_subplot(122, projection="3d")
ax.plot(x_simulate_run2[:, 0], x_simulate_run2[:, 1], x_simulate_run2[:, 2])
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.title("Identified System")
fig.show()

# %%
