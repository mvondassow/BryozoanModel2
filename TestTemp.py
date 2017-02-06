# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 22:13:33 2017

Tests on attempts to find a simpler way to solve for pressures in
bryozoan model.

@author: Michelangelo
"""

from Bryozoan import *
import numpy
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt

c1 = Colony(nz=6, mz=7, OutflowConductivity=0.01, dCdt=dCdt_default,
                 dCdt_in_params={'yminusx': 1, 'b': 1, 'r': 1, 'w': 3},                 
                 dCdt_out_params={'yminusx': 1, 'b': 0.1, 'r': 1, 'w': 3})
        # Set a central outflow conduit (edge) to have higher conductivity
c1.setouterconductivities([41], [0.02])

temp = c1.solvecolony()
Bz = temp['IncidenceFull']
P = np.asmatrix(temp['Pressures']).transpose()
C = sparse.diags(temp['conductivityfull'],0)
Cinv = sparse.diags(1/temp['conductivityfull'],0)
q = np.asmatrix(c1.InFlow).transpose()
mpiBz = scipy.linalg.pinv(Bz.todense())

q_newmethod = Bz.transpose()*C*Bz*(mpiBz*(Cinv*(mpiBz.transpose()*q)))
q_originalmethod = Bz.transpose()*C*Bz*P

print('original method works?', np.allclose(q, q_originalmethod, 
                                            rtol=np.mean(abs(q))*10**-4,
                                            atol=np.mean(abs(q))*10**-4))

print('new method works?', np.allclose(q, q_newmethod,
                                       rtol=np.mean(abs(q))*10**-4,
                                       atol=np.mean(abs(q))*10**-4))

print('Is it a right inverse?', 
      np.allclose(q, Bz.transpose()*mpiBz.transpose()*q,
                  rtol=np.mean(abs(q))*10**-4, atol=np.mean(abs(q))*10**-4))
      
print('Is it a right inverse?', 
      np.allclose(Bz.transpose()*C*Bz*P, Bz.transpose()*mpiBz.transpose()*q,
                  rtol=np.mean(abs(q))*10**-4, atol=np.mean(abs(q))*10**-4))

print('Is it a right inverse?', 
      np.allclose(C*Bz*P, mpiBz.transpose()*q,
                  rtol=np.mean(abs(q))*10**-4, atol=np.mean(abs(q))*10**-4))

print('Is it a right inverse?', 
      np.allclose(Bz*P, ssl.inv(C)*mpiBz.transpose()*q,
                  rtol=np.mean(abs(q))*10**-4, atol=np.mean(abs(q))*10**-4))
