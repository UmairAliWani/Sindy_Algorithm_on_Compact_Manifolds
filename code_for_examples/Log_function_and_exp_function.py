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
import sympy
from sympy.matrices import Matrix
from sympy import symbols
from sympy import zeros,diff,cos,simplify,lambdify

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
subs_dict = {R: 3, r: 1}
Gamma_Rr_eval = [[[Gamma[m][i][j].subs(subs_dict) for j in range(n)] for i in range(n)] for m in range(n)]
metric_tensor_Rr_eval=[[metric_tensor[i,j].subs(subs_dict) for j in range(n)] for i in range(n)]
inverse_metric_tensor_Rr_eval=[[inverse_metric_tensor[i,j].subs(subs_dict) for j in range(n)] for i in range(n)]

 # %%
 ## Converting symbolic functions into callable functions
Gamma_function=[[[lambdify((theta, phi), Gamma_Rr_eval[k][i][j]) for j in range(n)] for i in range(n)] for k in range(n)]
metric_tensor_function=[[lambdify((theta, phi),metric_tensor_Rr_eval[i][j]) for j in range(n)] for i in range(n)]
inverse_metric_tensor_function=[[lambdify((theta, phi),inverse_metric_tensor_Rr_eval[i][j]) for j in range(n)] for i in range(n)]
