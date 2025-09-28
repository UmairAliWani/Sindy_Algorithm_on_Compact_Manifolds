{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0fcef40f-136f-4a88-a09a-f4269bb89c6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sympy\n",
    "from sympy.matrices import Matrix\n",
    "from sympy import symbols\n",
    "from sympy import zeros,diff,cos,simplify,lambdify"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "542a2f21-5f0c-4ba6-8f55-1df78af82385",
   "metadata": {},
   "outputs": [],
   "source": [
    "theta=symbols('theta')\n",
    "phi=symbols('phi')\n",
    "R=symbols('R')\n",
    "r=symbols('r')\n",
    "coords=[theta,phi]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "dbaa87a7-8340-481a-aa39-739d6bc61c47",
   "metadata": {},
   "outputs": [],
   "source": [
    "##Defining the metric tensor and inverse metric tensor\n",
    "metric_tensor=Matrix([[(R + r*cos(phi))**2, 0], [0, r**2]])\n",
    "inverse_metric_tensor= metric_tensor.inv()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "917856f6-96f5-48db-a5b3-36bb07b3b025",
   "metadata": {},
   "outputs": [],
   "source": [
    "n=len(coords)\n",
    "Gamma=[[[0 for k in range(n)] for i in range(n)] for j in range(n)]  ###Initialize a list representing 3D array for christofle symbols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cbee14c6-53aa-441b-80f7-0c958915838e",
   "metadata": {},
   "outputs": [],
   "source": [
    "###Function for defining the christoffel symbols####\n",
    "def christofel_symbols(metric_tensor,inverse_metric_tensor):\n",
    "    for m in range(n):\n",
    "        for i in range(n):\n",
    "            for j in range(n):\n",
    "                sum=0\n",
    "                for l in range(n):\n",
    "                    dg_j_g_il=diff(metric_tensor[i,l],coords[j])\n",
    "                    dg_i_g_lj=diff(metric_tensor[l,j],coords[i])    ###Formula for getting Christoffel symbols\n",
    "                    dg_l_g_ji=diff(metric_tensor[j,i],coords[l])\n",
    "                    sum= sum + inverse_metric_tensor[m,l]*(dg_j_g_il + dg_i_g_lj - dg_l_g_ji)\n",
    "                Gamma[m][i][j]=simplify(0.5*sum)\n",
    "    return Gamma  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "3c7038de-195a-4803-8ef3-619e05b76f6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "Gamma=christofel_symbols(metric_tensor,inverse_metric_tensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "77ec0bba-69a5-4be8-a4b1-47c07a0c85e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "subs_dict = {R: 3, r: 1}\n",
    "Gamma_Rr_eval = [[[Gamma[m][i][j].subs(subs_dict) for j in range(n)] for i in range(n)] for m in range(n)]\n",
    "metric_tensor_Rr_eval=[[metric_tensor[i,j].subs(subs_dict) for j in range(n)] for i in range(n)]\n",
    "inverse_metric_tensor_Rr_eval=[[inverse_metric_tensor[i,j].subs(subs_dict) for j in range(n)] for i in range(n)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "2fdacb58-72d1-4b88-be24-5c4deeb4ff69",
   "metadata": {},
   "outputs": [],
   "source": [
    " ## Converting symbolic functions into callable functions\n",
    "Gamma_function=[[[lambdify((theta, phi), Gamma_Rr_eval[k][i][j]) for j in range(n)] for i in range(n)] for k in range(n)]\n",
    "metric_tensor_function=[[lambdify((theta, phi),metric_tensor_Rr_eval[i][j]) for j in range(n)] for i in range(n)]\n",
    "inverse_metric_tensor_function=[[lambdify((theta, phi),inverse_metric_tensor_Rr_eval[i][j]) for j in range(n)] for i in range(n)]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
