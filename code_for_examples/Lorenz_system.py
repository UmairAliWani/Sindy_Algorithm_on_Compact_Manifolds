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
poly_lib = PolynomialLibrary(degree=5)


# %%
def lorenz_system(t,X):
    x,y,z=X
    dxdt=10*(y-x)
    dydt=x*(28-z)-y
    dzdt=x*y-(8/3)*z
    return [dxdt, dydt, dzdt]



# %%
dt=0.001
t_span_train=np.arange(0,100,dt)
t_interval=(0,100)
X0=[-8,8,27]
X_train=solve_ivp(lorenz_system,t_interval,X0,t_eval=t_span_train, **integrator_keywords).y.T
X_dot_train=np.array([lorenz_system(0,X_train[i]) for i in range(t_span_train.size)])

# %%
lorenz_model=ps.SINDy(feature_library=poly_lib)
lorenz_model.fit(X_train,t=dt)
lorenz_model.print()

# %%
models=[]
X_sims=[]
X_tests=[]
noise=[0.01, 0.1, 1.0,10.0]
t_test=np.arange(0,20,dt)
t_nlinterval_test=(0,20)

# %%
for eps in noise:
    X_dot_trains= X_dot_train + np.random.normal(loc=0, scale=eps, size=X_dot_train.shape)
    model= ps.SINDy(feature_library=poly_lib)
    model.fit(X_train, t=dt, x_dot=X_dot_trains)
    models.append(model)
    X_sims.append(model.simulate(X0,t_test))

# %%
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(121, projection="3d")
ax.plot(X_train[:,0],X_train[:,1],X_train[:,2],'blue')
plt.title("True Simulation")
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")

ax = fig.add_subplot(122, projection="3d")
ax.plot(X_train[: t_test.size,0],X_train[: t_test.size,1],X_train[: t_test.size,2],'red')
ax.set_title('Model Simulation')
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.savefig("Lorenz_system3d_identifaction.png")

# %%
#Trained model
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot(X_train[:,0],X_train[:,1],X_train[:,2],'blue')
ax.set_title('Full simulation')
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.show()

# %%
#Trained model on test time.
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot(X_train[: t_test.size,0],X_train[: t_test.size,1],X_train[: t_test.size,2],'blue')
ax.set_title('Full simulation')
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.show()

# %%
#Identified model for eta=0.01
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot(X_sims[0][:,0],X_sims[0][:,1],X_sims[0][:,2],'blue')
ax.set_title('Identified system for $\eta =0.01$')
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.show()

# %%
#Identified model for eta=1
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot(X_sims[2][:,0],X_sims[2][:,1],X_sims[2][:,2],'blue')
ax.set_title('Identified system for $\eta =1.0$')
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.show()

# %%
fig, ax=plt.subplots()
ax.plot(t_test,X_sims[0][:,0],"k", label=r"$x_1$")
ax.plot(t_test,X_train[: t_test.size,0],"r--", label="model")
ax.legend()
plt.xlabel("t")
plt.ylabel(r"$x_1$")
fig.show()
plt.title("Model comparison for $\eta=0.01$")
plt.savefig("Lorenz_with_noise_x1.png")

# %%
fig, ax=plt.subplots()
ax.plot(t_test,X_sims[1][:,1],"k", label=r"$x_2$")
ax.plot(t_test,X_train[: t_test.size,1],"r--", label="model")
ax.legend()
plt.xlabel("t")
plt.ylabel(r"$x_2$")
fig.show()
plt.title("Model comparison for $\eta=0.01$")
plt.savefig("Lorenz_system_noise_x2.png")

# %%
colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

# %%
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot(X_train[: t_test.size,0],X_train[: t_test.size,1],X_train[: t_test.size,2],'blue')
ax.set_title('Full simulation')
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.show()

# %%
fig, ax=plt.subplots()
for i,x in enumerate(X_sims):
    ax.semilogy(t_test, np.sum((x-X_train[:t_test.size])**2,axis=1),color=colors[i],label=noise[i],)
    ax.legend(title="noise")
    plt.xlabel("t")
    plt.ylabel("Log error")
    fig.show()
    plt.title("Error for measured $\dot{x}$ and x")

# %%
