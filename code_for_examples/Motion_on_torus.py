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
def particle_on_torus(t, X, R=3.0, r=1.0,g=9.8):
    theta, phi, theta_dot, phi_dot = X
    dtheta_dt = theta_dot
    dphi_dt= phi_dot
    dtheta_dot_dt = -(R+r*np.cos(theta))*np.sin(theta)*dphi_dt**2/r - g*np.cos(theta)/r
    dphi_dot_dt=2*r*np.sin(theta)/(R +r*np.cos(theta))
    return [dtheta_dt, dphi_dt, dtheta_dot_dt,dphi_dot_dt]
X0 = [np.pi/4, np.pi/3, 0.9, 0.9]
dt = 0.001
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
X_train = solve_ivp(particle_on_torus, t_train_span, X0, t_eval=t_train)
theta = X_train.y[0]
phi = X_train.y[1]
theta_dot = X_train.y[2]
phi_dot=X_train.y[3]

# %%
theta_ddot = np.gradient(theta_dot, dt)
phi_ddot=np.gradient(phi_dot, dt)

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
   # lambda theta, phi, theta_dot, phi_dot:np.sin(theta)* np.cos(theta),
    #lambda theta, phi, theta_dot, phi_dot: 1 / np.tan(theta),  # cot(θ)

    # Physics-informed terms
    lambda theta, phi, theta_dot, phi_dot: np.sin(theta) * phi_dot**2,
   # lambda theta, phi, theta_dot, phi_dot: np.cos(theta) * phi_dot**2,
    lambda theta, phi, theta_dot, phi_dot:np.sin(theta)* np.cos(theta)*phi_dot**2,
    lambda theta, phi, theta_dot, phi_dot: np.sin(theta)/(3+ np.cos(theta)),
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
   # "sin(θ)·cos(θ)",
    #"cot(θ)",
    #"sin(θ)·φ̇",
     "sin(θ)·φ̇²",
   # "cos(θ)·φ̇²",
    "sin(θ)·cos(θ)·φ̇²",
    "sin(θ)/(3+ cos(θ))",
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
def particle_on_torus_sim(t, X, R=3.0, r=1.0,g=9.8):
    theta, phi, theta_dot, phi_dot = X
    dtheta_dt = theta_dot
    dphi_dt= phi_dot
    dtheta_dot_dt = -(3.0266 + 0.9775*np.cos(theta))*np.sin(theta)*dphi_dt**2 - 9.8180*np.cos(theta)
    dphi_dot_dt=2.0038*np.sin(theta)/(3 + np.cos(theta))
    return [dtheta_dt, dphi_dt, dtheta_dot_dt,dphi_dot_dt]


# %%
X0 = [np.pi/4, np.pi/3, 0.9, 0.9]
dt = 0.001
t_sim = np.arange(0, 9, dt)
t_sim_span = (t_sim[0], t_sim[-1])
X_simulated = solve_ivp(particle_on_torus_sim, t_sim_span, X0, t_eval=t_sim)
theta_mod=X_simulated.y[0]
phi_mod=X_simulated.y[1]
thetadot_mod=X_simulated.y[2]
phidot_mod=X_simulated.y[3]

# %%
##I am renaming the variables here because second example has been done using new variables. So, as to keep consistency in pdf file, I will
#theta as phi and phi as theta 

fig,ax=plt.subplots()
ax.plot(t_sim, theta[:t_sim.size], "c", label=r" True $\phi$")
ax.plot(t_sim,theta_mod,"r--",label=r"Model $\phi$")
ax.legend()
ax.set_xlabel("t")  
ax.set_ylabel(r"$\phi$")
fig.show()
plt.savefig("Torus_phi_traject.png")

# %%
fig,ax=plt.subplots()
ax.plot(t_sim, phi[:t_sim.size], "c", label=r"True $\theta$")
ax.plot(t_sim,phi_mod,"r--",label=r"Model $\theta$")
ax.legend()
ax.set_xlabel("t")  
ax.set_ylabel(r"$\theta$")
fig.show()
plt.savefig("Torus_theta_traj.png")

# %%
fig,ax=plt.subplots()
ax.plot(t_sim, theta_dot[:t_sim.size], "c", label=r"True $\dot{\phi}$")
ax.plot(t_sim,thetadot_mod,"r--",label=r"Model $\dot{\phi}$")
ax.legend()
ax.set_xlabel("t")  
ax.set_ylabel(r"$\dot{\phi}$")
fig.show()
plt.savefig("Torus_dot_phi_traject.png")

# %%
fig,ax=plt.subplots()
ax.plot(t_sim, phi_dot[:t_sim.size], "c", label=r" True $\dot{\theta}$")
ax.plot(t_sim,phidot_mod,"r--",label=r"Model $\dot{\theta}$")
ax.legend()
ax.set_xlabel("t")  
ax.set_ylabel(r"$\dot{\theta}$")
fig.show()
plt.savefig("Torus_dot_theta_traject.png")

# %%
x=(3+np.cos(theta))*np.cos(phi);
y=(3+np.cos(theta))*np.sin(phi);
z=np.sin(theta);
x_sim=(3+np.cos(theta_mod))*np.cos(phi_mod);
y_sim=(3+np.cos(theta_mod))*np.sin(phi_mod);
z_sim=np.sin(theta_mod);

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
plt.savefig("Torus_trained_identified_systems.png")
