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
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy
from sympy.matrices import Matrix
from sympy import symbols
from sympy import zeros,diff,cos,simplify,lambdify
from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
class SINDyPI:
    def __init__(self, threshold=1e-5, max_iter=1000, alpha=1e-6):
        self.threshold = threshold
        self.max_iter = max_iter
        self.alpha = alpha
        self.coef_ = None
        self.feature_names_ = None

    def fit(self, Theta, y):
        """
        Fit the implicit equation: Theta @ xi ≈ y (usually y ≈ 0 for implicit discovery).
        """
        # Remove near-zero columns (reduces instability)
        Theta = np.array(Theta)
        col_norms = np.linalg.norm(Theta, axis=0)
        nonzero_cols = col_norms > self.threshold
        self._col_mask = nonzero_cols
        Theta_reduced = Theta[:, nonzero_cols]

        # # Fit using Lasso for sparsity
        # lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter)
        # lasso.fit(Theta_reduced, y)

     #######  # Fit using STLSQ#############
        lr=LinearRegression(fit_intercept=False)
        x=lr.fit(Theta_reduced, y).coef_
        for _ in range(self.max_iter):
            small_coeff = np.abs(x) < self.threshold
            if np.all(small_coeff):
                break

            x[small_coeff] = 0
            big_coeffs = ~small_coeff
            if not np.any(big_coeffs):
                break
            x[big_coeffs] = lr.fit(Theta_reduced[:, big_coeffs], y).coef_

        # Restore to full-size coefficient vector
        coef_full = np.zeros(Theta.shape[1])
        coef_full[nonzero_cols] = x
        self.coef_ = coef_full
        return self

        # # Store back in full-sized coefficient vector
        # coef_full = np.zeros(Theta.shape[1])
        # coef_full[nonzero_cols] = lasso.coef_
        # self.coef_ = coef_full
        # return self

    def coefficients(self, feature_names=None):
        """
        Return a dict of nonzero coefficients for readability.
        """
        self.feature_names_ = feature_names
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(self.coef_))]
        return {
            name: coef for name, coef in zip(feature_names, self.coef_)
            if abs(coef) > self.threshold
        }


# %% [markdown]
# I will first of all embedd the torus in $R^3$. The embedding is given as:
# $$x=(R+r \cos\phi) \cos \theta$$
# $$y=(R+r \cos\phi) \sin \theta$$
# $$z= r \sin \phi$$
# The metric is then given as:
# $$\begin{pmatrix} (R+r\cos\phi)^2 & 0\\ 0 & r^2 \end{pmatrix}$$

# %%
theta=symbols('theta')
phi=symbols('phi')
R=symbols('R')
r=symbols('r')
coords=[theta,phi]

# %%
##Defining the metric tensor and inverse metric tensor
metric_tensor=Matrix([[(R + r*cos(phi))**2, 0], [0, r**2]])
inverse_metric_tensor= metric_tensor.inv()

# %%
n=len(coords)
Gamma=[[[0 for k in range(n)] for i in range(n)] for j in range(n)]  ###Initialize a list representing 3D array for christofle symbols


 # %%
 ###Function for defining the christoffel symbols####
def christofel_symbols(metric_tensor,inverse_metric_tensor):
    for m in range(n):
        for i in range(n):
            for j in range(n):
                sum=0
                for l in range(n):
                    dg_j_g_il=diff(metric_tensor[i,l],coords[j])
                    dg_i_g_lj=diff(metric_tensor[l,j],coords[i])    ###Formula for getting Christoffel symbols
                    dg_l_g_ji=diff(metric_tensor[j,i],coords[l])
                    sum= sum + inverse_metric_tensor[m,l]*(dg_j_g_il + dg_i_g_lj - dg_l_g_ji)
                Gamma[m][i][j]=simplify(0.5*sum)
    return Gamma   


# %%
Gamma=christofel_symbols(metric_tensor,inverse_metric_tensor)

  # %%
  ##Fixing the radii for the torus###################
subs_dict = {R: 3, r: 1}
Gamma_Rr_eval = [[[Gamma[m][i][j].subs(subs_dict) for j in range(n)] for i in range(n)] for m in range(n)]
metric_tensor_Rr_eval=[[metric_tensor[i,j].subs(subs_dict) for j in range(n)] for i in range(n)]
inverse_metric_tensor_Rr_eval=[[inverse_metric_tensor[i,j].subs(subs_dict) for j in range(n)] for i in range(n)]

 # %%
 ## Converting symbolic functions into callable functions
Gamma_function=[[[lambdify((theta, phi), Gamma_Rr_eval[k][i][j]) for j in range(n)] for i in range(n)] for k in range(n)]
metric_tensor_function=[[lambdify((theta, phi),metric_tensor_Rr_eval[i][j]) for j in range(n)] for i in range(n)]
inverse_metric_tensor_function=[[lambdify((theta, phi),inverse_metric_tensor_Rr_eval[i][j]) for j in range(n)] for i in range(n)]


# %%
###Getting the training data###

def compact_manifold_torus(t,X):
    theta, phi, dtheta, dphi= X
    dx = [dtheta, dphi]
    ddx = [0, 0]
    g=9.81
    potential_term=[0,inverse_metric_tensor_function[1][1](theta,phi)*g*np.cos(phi)]
    
    for k in range(2):
        sum_gamma = 0
        for i in range(2):
            for j in range(2):
                gamma_val = Gamma_function[k][i][j](theta, phi)
                sum_gamma += gamma_val * dx[i] * dx[j]
        ddx[k] = -sum_gamma-potential_term[k] #+ np.random.normal(loc=0.0, scale=0.5)
    
    return [dtheta, dphi, ddx[0], ddx[1]]  


# %%
X0 = [np.pi/4, np.pi/3, 1 , 1]
dt = 0.001
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
X_train = solve_ivp(compact_manifold_torus, t_train_span, X0, t_eval=t_train)
theta = X_train.y[0]
phi = X_train.y[1]
theta_dot = X_train.y[2]
phi_dot=X_train.y[3]


# %%
###Defining the geodesic equation###
def exponential_system(t,X):
    theta, phi, dtheta, dphi= X
    dx = [dtheta, dphi]
    ddx = [0, 0]
    g=9.81
    potential_term=[0,inverse_metric_tensor_function[1][1](theta,phi)*g*np.cos(phi)]
    
    for k in range(2):
        sum_gamma = 0
        for i in range(2):
            for j in range(2):
                gamma_val = Gamma_function[k][i][j](theta, phi)
                sum_gamma += gamma_val * dx[i] * dx[j]
        ddx[k] = -sum_gamma-potential_term[k] #+ np.random.normal(loc=0.0, scale=0.5)
    
    return [dtheta, dphi, ddx[0], ddx[1]]  


# %%
def exponential_function(ini):
    ini= [theta[0], phi[0], theta_dot[0] ,phi_dot[0] ]
    dt = 0.001
    t_geodesic = np.arange(0, 1, dt)
    t_geodesic_span = (t_geodesic[0], t_geodesic[-1])
    X_geo = solve_ivp(exponential_function, t_geodesic_span, ini, t_eval=t_geodesic)
    return X_geo.y[0:2, -1]


# %%
exp_val=exponential_function(exponential_system)

# %%
theta[0]

# %%
ini_vect=np.concatenate([[theta[0]], [phi[0]]])
