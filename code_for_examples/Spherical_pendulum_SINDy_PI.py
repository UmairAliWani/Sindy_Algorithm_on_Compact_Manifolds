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


# %%
def spherical_pendulum(t, X, L=1.0, g=9.81):
    theta, phi, theta_dot, phi_dot = X
    dtheta_dt = theta_dot
    dphi_dt = phi_dot
    dtheta_dot_dt = (np.sin(theta) * np.cos(theta) * phi_dot**2) - (g / L) * np.sin(theta)
    dphi_dot_dt = -2 * (theta_dot * phi_dot) / np.tan(theta)
    return [dtheta_dt, dphi_dt, dtheta_dot_dt, dphi_dot_dt]
X0 = [np.pi/2, np.pi/2, 0.5, 0.5]
dt = 0.001
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
X_train = solve_ivp(spherical_pendulum, t_train_span, X0, t_eval=t_train)
theta = X_train.y[0]
phi = X_train.y[1]
theta_dot = X_train.y[2]
phi_dot = X_train.y[3]

# %%
theta_ddot = np.gradient(theta_dot, dt)
phi_ddot = np.gradient(phi_dot, dt)

# %%
library_functions = [
    # Base terms: single-variable
    lambda theta, phi, theta_dot, phi_dot: theta,
    lambda theta, phi, theta_dot, phi_dot: phi,
    lambda theta, phi, theta_dot, phi_dot: theta_dot,
    lambda theta, phi, theta_dot, phi_dot: phi_dot,

    # Quadratic terms
    #lambda theta, phi, theta_dot, phi_dot: theta**2,
    #lambda theta, phi, theta_dot, phi_dot: phi**2,
    #lambda theta, phi, theta_dot, phi_dot: theta_dot**2,
    #lambda theta, phi, theta_dot, phi_dot: phi_dot**2,

    # Interaction terms
   # lambda theta, phi, theta_dot, phi_dot: theta * phi,
    #lambda theta, phi, theta_dot, phi_dot: theta * theta_dot,
    #lambda theta, phi, theta_dot, phi_dot: theta * phi_dot,
    lambda theta, phi, theta_dot, phi_dot: theta_dot * phi_dot,

    # Trigonometric terms
    lambda theta, phi, theta_dot, phi_dot: np.sin(theta),
    lambda theta, phi, theta_dot, phi_dot: np.cos(theta),
    lambda theta, phi, theta_dot, phi_dot:np.sin(theta)* np.cos(theta),
    #lambda theta, phi, theta_dot, phi_dot: 1 / np.tan(theta),  # cot(θ)

    # Physics-informed terms
    lambda theta, phi, theta_dot, phi_dot: np.sin(theta) * phi_dot,
    lambda theta, phi, theta_dot, phi_dot: np.cos(theta) * phi_dot**2,
    lambda theta, phi, theta_dot, phi_dot:np.sin(theta)* np.cos(theta)*phi_dot**2,
    lambda theta, phi, theta_dot, phi_dot: theta_dot * phi_dot / np.tan(theta),
]

# --- Corresponding feature names ---
library_function_names = [
    "θ", "φ",
    "θ̇", "φ̇",
    #"θ²", "φ²",
    #"θ̇²",
    #"φ̇²",
  #  "θ·φ", "θ·θ̇", "θ·φ̇",
    "θ̇·φ̇",
    "sin(θ)", "cos(θ)",
    "sin(θ)·cos(θ)",
    #"cot(θ)",
    "sin(θ)·φ̇",
    "cos(θ)·φ̇²",
    "sin(θ)·cos(θ)·φ̇²",
    "θ̇·φ̇·cot(θ)",
]


# %%
##Generating the library
Theta = np.column_stack([
    f(theta, phi, theta_dot, phi_dot) for f in library_functions
])

# %%
model_theta = SINDyPI(threshold=0.6, alpha=1e-6)
model_theta.fit(Theta, theta_ddot)
theta_coeffs = model_theta.coefficients(library_function_names)

# %%
model_phi = SINDyPI(threshold=0.6, alpha=1e-6)
model_phi.fit(Theta, phi_ddot)
phi_coeffs = model_phi.coefficients(library_function_names)


# %%
def print_equation(name, coeffs):
    print(f"\n{name} ≈ ", end="")
    terms = [f"{coef:+.4f}·{fname}" for fname, coef in coeffs.items()]
    print(" ".join(terms))


# %%
print_equation("θ̈", theta_coeffs)

# %%
print_equation("φ̈", phi_coeffs)

# %%
theta_coeffs


# %%
def spherical_pendulum_modelled(t, X):
    theta, phi, theta_dot, phi_dot = X
    dtheta_dt = theta_dot
    dphi_dt = phi_dot
    dtheta_dot_dt = 0.9980*(np.sin(theta) * np.cos(theta) * phi_dot**2) - 9.8160* np.sin(theta)
    dphi_dot_dt = -2.0004 * (theta_dot * phi_dot) / np.tan(theta)
    return [dtheta_dt, dphi_dt, dtheta_dot_dt, dphi_dot_dt]


# %%
X0 = [np.pi/2, np.pi/2, 0.5, 0.5]
dt = 0.001
t_sim = np.arange(0, 9, dt)
t_sim_span = (t_sim[0], t_sim[-1])
X_simulated = solve_ivp(spherical_pendulum_modelled, t_sim_span, X0, t_eval=t_sim)
theta_mod=X_simulated.y[0]
phi_mod=X_simulated.y[1]
thetadot_mod=X_simulated.y[2]
phidot_mod=X_simulated.y[3]

# %%
fig,ax=plt.subplots()
ax.plot(t_sim, theta[:t_sim.size], "c", label=r"True $\theta$")
ax.plot(t_sim,theta_mod,"r--",label=r"Model $\theta$")
ax.legend()
ax.set_xlabel("t")  
ax.set_ylabel(r"$\theta$")
fig.show()
plt.savefig("Spherical_pendulum_theta.png")

# %%
fig,ax=plt.subplots()
ax.plot(t_sim, phi[:t_sim.size], "c", label=r" True $\phi$")
ax.plot(t_sim,phi_mod,"r--",label=r"Model $\phi$")
ax.legend()
ax.set_xlabel("t")  
ax.set_ylabel(r"$\phi$")
fig.show()
plt.savefig("Spherical_pendulum_phi.png")

# %%
fig,ax=plt.subplots()
ax.plot(t_sim, theta_dot[:t_sim.size], "c", label=r"True $\dot{\theta}$")
ax.plot(t_sim,thetadot_mod,"r--",label=r"Model $\dot{\theta}$")
ax.legend()
ax.set_xlabel("t")  
ax.set_ylabel(r"$\dot{\theta}$")
fig.show()
plt.savefig("Spherical_pendulum_theta_dot.png")

# %%
fig,ax=plt.subplots()
ax.plot(t_sim, phi_dot[:t_sim.size], "c", label=r" True $\dot{\phi}$")
ax.plot(t_sim,phidot_mod,"r--",label=r"Model $\dot{\phi}$")
ax.legend()
ax.set_xlabel("t")  
ax.set_ylabel(r"$\dot{\phi}$")
fig.show()
plt.savefig("Spherical_pendulum_phi_dot.png")

# %%
x=np.sin(theta)*np.cos(phi);
y=np.sin(theta)*np.sin(phi);
z=np.cos(theta);
x_sim=np.sin(theta_mod)*np.cos(phi_mod);
y_sim=np.sin(theta_mod)*np.sin(phi_mod);
z_sim=np.cos(theta_mod);

# %%
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(121, projection="3d")
ax.plot(x, y, z)
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.title("Trained System")

ax = fig.add_subplot(122, projection="3d")
ax.plot(x_sim, y_sim, z_sim)
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
plt.title("Identified System")
plt.savefig("spherical_pendulum_identify_trained_system.png")

# %%
