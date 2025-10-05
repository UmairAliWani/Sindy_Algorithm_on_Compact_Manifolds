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
integrator_keywords["rtol"] = 1e-6
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-6


# %%
def two_d_linear_damped(t,Y):
    Y1,Y2=Y
    dY1dt=-0.1*Y1 +2*Y2
    dY2dt=-2*Y1-0.1*Y2
    return [dY1dt, dY2dt]


# %%
def two_d_nonlinear_damped(t,Y):
    Y1,Y2=Y
    dY1dt=-0.1*(Y1**3) + 2*(Y2**3)
    dY2dt=-2*(Y1**3)-0.1*(Y2**3)
    return [dY1dt, dY2dt]


# %%
dt=0.01

#Training the data
t_span_train=np.arange(0,1,dt)
t_interval=(0,1)
Y0=[0,2]
Y_train=solve_ivp(two_d_linear_damped,t_interval,Y0,t_eval=t_span_train, **integrator_keywords).y.T

# %%
twod_damp_model=ps.SINDy()
twod_damp_model.fit(Y_train, t=dt)
twod_damp_model.print()

# %%
t_span_test=np.arange(0,25,dt)
t_interval_test=(0,25)
Y0_test=[2,0]
Y_test=solve_ivp(two_d_linear_damped,t_interval_test,Y0_test,t_eval=t_span_test, **integrator_keywords).y.T

# %%
Y_sim=twod_damp_model.simulate(Y0_test,t_span_test)

# %%
fig, ax=plt.subplots()
ax.plot(t_span_test,Y_test[:,0],"k", label=r"$x_1$")
ax.plot(t_span_test,Y_sim[:,0],"r--", label="model")
ax.plot(t_span_test,Y_test[:,1],"c", label=r"$x_2$")
ax.plot(t_span_test,Y_sim[:,1],"b--", label="model")
ax.legend()
plt.xlabel("t")
plt.ylabel(r"$x_k$")
fig.show()
plt.title("Linear System")
plt.savefig("Linear_damp_osc.png")

# %%
fig, ax=plt.subplots()
ax.plot(Y_sim[:,0],Y_sim[:,1],"c", label=r"$x_k$")
ax.plot(Y_test[:,0],Y_test[:,1],"r--",label="model")
ax.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
fig.show()
plt.title("Phase Diagram")
plt.savefig("Linear_damp_osc_phase_diag.png")

# %%
library_functions=[
    lambda x : x,
    lambda x : x**2,
    lambda x : x**3,
]
library_function_names=[
    lambda x : f"{x}",
    lambda x : f"{x}^2",
    lambda x: f"{x}^3",
]

# %%
custom_library = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names)

# %%
dt_nonlinear=0.01
t_nlspan_train=np.arange(0,1,dt_nonlinear)
t_nlinterval=(0,1)
Y0_nonlinear=[2,0]
Y_train_nonlinear=solve_ivp(two_d_nonlinear_damped,t_nlinterval,Y0_nonlinear,t_eval=t_nlspan_train, **integrator_keywords).y.T

# %%
stlsq_optimizer = ps.STLSQ(threshold=0.05)
twod_nonlinear_damp_model=ps.SINDy(feature_library=custom_library,optimizer=stlsq_optimizer)
twod_nonlinear_damp_model.fit(Y_train_nonlinear, t=dt_nonlinear)
twod_nonlinear_damp_model.print()

# %%
t_nlspan_test=np.arange(0,25,dt)
t_nlinterval_test=(0,25)
Y0_nltest=[2,0]
Y_nltest=solve_ivp(two_d_nonlinear_damped,t_nlinterval_test,Y0_nltest,t_eval=t_nlspan_test, **integrator_keywords).y.T

# %%
Y_nlsim=twod_nonlinear_damp_model.simulate(Y0_nltest,t_nlspan_test)

# %%
fig, ax=plt.subplots()
ax.plot(t_nlspan_test,Y_nltest[:,0],"k", label=r"$x_1$")
ax.plot(t_nlspan_test,Y_nlsim[:,0],"r--", label="model")
ax.plot(t_nlspan_test,Y_nltest[:,1],"c", label=r"$x_2$")
ax.plot(t_nlspan_test,Y_nlsim[:,1],"b--", label="model")
ax.legend()
plt.xlabel("t")
plt.ylabel(r"$x_k$")
fig.show()
plt.title("Cubic System")
plt.savefig("Cubic_systm_damped_osc.png")

# %%
fig, ax=plt.subplots()
ax.plot(Y_nlsim[:,0],Y_nlsim[:,1],"c", label=r"$x_k$")
ax.plot(Y_nltest[:,0],Y_nltest[:,1],"r--",label="model")
ax.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
fig.show()
plt.title("Phase Diagram")
plt.savefig("Cubic_systm_damped_osc_phase_daig.png")

# %%
