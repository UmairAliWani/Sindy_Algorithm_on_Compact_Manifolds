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
import numpy as np
import pysindy as ps


# %%
t=np.linspace(0,1,100)
x=3*np.exp(-2*t)
y=0.5*np.exp(t)
X=np.stack((x,y),axis=-1);

# %%
model=ps.SINDy(feature_names=["x","y"])
model.fit(X,t=t)

# %%
model.print()

# %%
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
from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control
if __name__ != "testing":
    t_end_train = 10
    t_end_test = 15
else:
    t_end_train = 0.04
    t_end_test = 0.04

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
np.random.seed(100)

# %%
# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

# %%
# Generate measurement data
dt = 0.002

t_train = np.arange(0, t_end_train, dt)
x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T


 # %%
 #Instantiate and fit the SINDy model
model = ps.SINDy()
model.fit(x_train, t=dt)
model.print()


# %% [markdown]
# $\textbf{Example of damped oscillator}$
# $$M \frac {d^2y}{dt^2}+R \frac{dy}{dt}+\frac{\lambda y}{l}=0$$
# Above ODE as a system of first order ODEs is written as:
# $$\frac{dY_1}{dt}=Y_2$$
# $$\frac{dY_2}{dt}=-\frac{\lambda }{lM} Y_1-\frac{R}{M} Y_2$$
# Choose $M=2$, $R=3$, $l=1$, $\lambda=1$ with $y(0)=1$ and $\dot{y} (0)=1$.
# Solution is then given as:
# $$y(t)=-3e^{-t}+4e^{-\frac{t}{2}}$$
# Also, the SINDy algorithm should give the following:
# $$\dot{y_1}=1.00 y_2$$
# $$\dot{y_2}=-0.5 y_1 -1.5 y_2$$

# %%
def damped_oscillator(t,Y):
    Y1,Y2=Y
    M=2
    R=3
    l=1
    lam=1
    dY1dt=Y2
    dY2dt=-R/M *Y2 - lam/(l*M) *Y1
    return [dY1dt, dY2dt]


# %%
#Training the model for two solvers RK4 and LSODA
dt_damped=0.001
t_eval_train=np.arange(0,2,dt_damped)
t_span_train=(0,2)
ini_values_train=[0.2,0.3]  #0.2,0.3
Y_train_RK=solve_ivp(damped_oscillator,t_span_train,ini_values_train,t_eval=t_eval_train)
Y_train_lsoda=solve_ivp(damped_oscillator,t_span_train,ini_values_train,t_eval=t_eval_train, **integrator_keywords
).y.T

# %%
Y_train_new_RK=np.stack((Y_train_RK.y[0],Y_train_RK.y[1]),axis=-1) #Reshaping our solution from RK

# %%
RK_damped_model=ps.SINDy()
lsoda_damped_model=ps.SINDy()

# %%
RK_damped_model.fit(Y_train_new_RK, t=dt_damped)
lsoda_damped_model.fit(Y_train_lsoda, t=dt_damped)

# %%
RK_damped_model.print()

# %%
lsoda_damped_model.print()

# %%
#Testing our models
t_test=np.arange(0,1,dt_damped)
t_span_test=(0,1)
Y0_test=[1,2]
Y_test_lsoda=solve_ivp(damped_oscillator,t_span_test,Y0_test,t_eval=t_test, **integrator_keywords
).y.T
Y_test_RK=solve_ivp(damped_oscillator,t_span_test,Y0_test,t_eval=t_test)

# %%
Y_test_new_RK=np.stack((Y_test_RK.y[0],Y_test_RK.y[1]),axis=-1) #Reshaping our solution from RK

# %%
# Compare SINDy-predicted derivatives with finite difference derivatives
print("Model score: %f" % lsoda_damped_model.score(Y_test_lsoda, t=dt_damped))

# %%
print("Model score: %f" % RK_damped_model.score(Y_test_new_RK, t=dt_damped))

# %%
# Predict derivatives using the learned model
Y_dot_test_predicted_lsoda = lsoda_damped_model.predict(Y_test_lsoda)

# Compute derivatives with a finite difference method, for comparison
Y_dot_test_computed_lsoda = lsoda_damped_model.differentiate(Y_test_lsoda, t=dt_damped)

fig, axs = plt.subplots(Y_test_lsoda.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(Y_test_lsoda.shape[1]):
    axs[i].plot(t_test, Y_dot_test_computed_lsoda[:, i], "k", label="numerical derivative")
    axs[i].plot(t_test, Y_dot_test_predicted_lsoda[:, i], "r--", label="model prediction")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
fig.show()
fig.suptitle("Derivatives for LSODA Model", fontsize=14)

# %%
# Predict derivatives using the learned model
Y_dot_test_predicted_RK = RK_damped_model.predict(Y_test_new_RK)

# Compute derivatives with a finite difference method
Y_dot_test_computed_RK = RK_damped_model.differentiate(Y_test_new_RK, t=dt_damped)

fig, axs = plt.subplots(Y_test_new_RK.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(Y_test_new_RK.shape[1]):
    axs[i].plot(t_test, Y_dot_test_computed_RK[:, i], "k", label="numerical derivative")
    axs[i].plot(t_test, Y_dot_test_predicted_RK[:, i], "r--", label="model prediction")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
fig.show()
fig.suptitle("Derivatives for RK Model", fontsize=14)

# %%
#Simulating from the model
Y_sim_RK=RK_damped_model.simulate(Y0_test,t_test)

# %%
Y_sim_lsoda=lsoda_damped_model.simulate(Y0_test,t_test); 

# %%
fig,axs = plt.subplots(Y_test_new_RK.shape[1],1,sharex=True,figsize=(7,9))
for i in range(Y_test_new_RK.shape[1]):
    axs[i].plot(t_test, Y_test_new_RK[:, i], "k", label="true simulation")
    axs[i].plot(t_test, Y_sim_RK[:, i], "r--", label="model simulation")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))
fig.show()
fig.suptitle("Simulations for RK Model", fontsize=14)

# %%
fig,axs = plt.subplots(Y_test_lsoda.shape[1],1,sharex=True,figsize=(7,9))
for i in range(Y_test_new_RK.shape[1]):
    axs[i].plot(t_test, Y_test_lsoda[:, i], "k", label="true simulation")
    axs[i].plot(t_test, Y_sim_lsoda[:, i], "r--", label="model simulation")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))
fig.show()
fig.suptitle("Simulations for LSODA Model", fontsize=14)

# %%
