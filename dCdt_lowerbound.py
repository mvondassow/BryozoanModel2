# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 13:25:42 2017

dCdt function with lower bound on C for use by Colony class.

@author: Michelangelo
"""
from Bryozoan import Colony
import numpy as np
import copy
import time


def dCdt_lb(Cs, dPs, params):
    """
    Calculate dConductivity/dt (dC/dt) and S ('shear-like') quantifier.

    S captures the notion that large conduits should
    carry more flow than small conduits. It should be monotonically
    increasing function of flow (current, flux, etc) and monotonically
    decreasing function of conductivity. Shear (e.g. in blood vessels) is
    one example analogized here:

    Parameters
    ----------
    Cs : array, dim = 1
        1-by-n array of conductivities (floats)
    dPs : array, dim = 1
        1-by-n array of pressure differences (floats; same length as Cs)
    params : dictionary
        params must contain keys 'w', 'yminusx', 'r', 'b', 'c0' 
        for parameters (all numeric types).

    Returns
    -------
    tuple : length 2, first part is an array of values for dConductivity/dt,
        second part is an array of values for quantifier S. Both arrays are
        have 1 dimension, same length as input arrays.

    Justification for form of S & dC/dt:
    ------------------------------------
    shear*perimeter*length = pressureDrop*crosssectionArea

    h = parameter with units of length (radius, width, height...)
        describing conduit width

    perimeter ~ A*h^x, 0 ≤ x ≤ 1 (x = 0 if h is separation between infinite
        parallel plates; x = 1 if h is radius or width of conduit that
        scales isotropically, e.g. radius of cylindrical pipe).
    crosssectionArea ~ K*h^y, 1 ≤ y ≤ 2 (y = 1 if h is separation between
        infinite parallel plates; y = 2 if h is radius or width of conduit
        that scales isotropically (e.g. radius of cylindrical pipe);
        BUT, for height of vertical parallel plates, area and perimeter
        both x and y increase in direct proporion to height: y = x = 1.
    Therefore, assuming conduit length is constant:
    shear ~ pressureDrop*a*h^(y-x) : 0 ≤ (y-x) ≤ 1 (with a = K/(A*length)).

    conductivity ~ d*h^w, 1 ≤ w ≤ 4 (w = 3 if h is separation of plates;
        w = 4 if h is radius of cylindrical pipe; w = 1 if h is height of
        parallel vertical plates (assuming their separation is much smaller
        than their height)).

    Hence: h ~ (conductivity/d)^(1/w) and:
    shear ~ pressureDrop*(a/(d^z))*conductivity^z; z=(y-x)/w, b=a/(d^z)

    For the three cases above:
        separation between parallel plates: x = 0, y = 1, w = 3 : z = 1/3
        height of vertical parallel plates: x = 1, y = 1, w = 1 : z = 0
        radius of cylinder:                 x = 1, y = 2, w = 4 : z = 1/4

    Therefore, set S = b*(conductivity^z)*pressureDrop, 0 ≤ z ≤ 1/3
    As matrices (conductivity as diagonal matrix)
        S = b*abs(sum(conductivity[i,j]^z*sum(Incidence[j,k]*Pressures[k])

    Assuming dh/dt=r1*(S-s0), dC/dt = (d*h^(w-1))*dh/dt
    dC/dt = d*(C/d)^((w-1)/w)*r*(S-s0)
    dC/dt = d^((2w-1)/w) * C^((w-1)/w))*r(S-s0) can parameters such that:
    dC/dt = r*(C^q)*(S-1) with q = (w-1)/w so 0<q<3/4
    """
    w = params.get('w')
    z = params.get('yminusx')/w
    # Floor on conductivities set to 0.
    Cflr = np.maximum(Cs, 0)
    S = abs(params.get('b')*(Cflr**z)*dPs)
    dCdt = params.get('r') * (Cflr**((w-1)/w)) * (S - 1)
    # For Cs <= c0, only allow positive dC/dt
    dCdt[(Cflr < params.get('c0')) & (dCdt < 0)] = 0
    return dCdt, S

# It seems hard to figure out what's going on in this form:
# may be easier to understand model in terms of C and flow.

# Define initial colony
c1 = Colony(nz=7, mz=6, OutflowConductivity=0.001, dCdt=dCdt_lb,
                    dCdt_in_params={'yminusx': 1, 'b': 3, 'r': 0.2, 'w': 3,
                                    'c0': 0.5},
                    dCdt_out_params={'yminusx': 1, 'b': 0.3, 'r': 1, 'w': 3,
                                     'c0': 0.0009})
# Set a central outflow conduit (edge) to have higher conductivity
#c1.setouterconductivities([76], [0.02])
c1.setouterconductivities(list(range(70,84)), [1]*14)
c1.setouterconductivities([76], [2])

# Solve dif. eqs. for c1 and put result in c2.
t = time.time()
c2 = c1.develop(100)
print(time.time()-t)

# Perturb colony c2 by opening up outflow in lower left, then plot.
c3 = copy.deepcopy(c2)
c3.setouterconductivities([6], [c2.OutflowConduits.max()])

# Solve dif. eqs. for c2 after perturbation, put in c3 & plot.
t = time.time()
c4 = c3.develop(100)
print(time.time()-t)

# Parameters for plotting
lsc = 0.9  # Scale lines for edges connecting inner nodes
dsc = 120  # Scale marker for edges from inner nodes to outer node (stars)
osc = 15  # Scale markers for flow from inner nodes to outer node
isc = 160  # Scale arrows for flow between inner nodes.
lpw = 1/2  # Exponent for converting inner conduit conductivity to line widths
dpw = 2/3  # Exponent for converting outflow conduit conductivity to dot areas

# Create plots
c1.colonyplot(False, linescale=lsc, dotscale=dsc, outflowscale=osc,
              innerflowscale=isc, linepwr=lpw, dotpwr=dpw)
c2.colonyplot(False, linescale=lsc, dotscale=dsc, outflowscale=osc,
              innerflowscale=isc, linepwr=lpw, dotpwr=dpw)
c3.colonyplot(False, linescale=lsc, dotscale=dsc, outflowscale=osc,
              innerflowscale=isc, linepwr=lpw, dotpwr=dpw)
c4.colonyplot(False, linescale=lsc, dotscale=dsc, outflowscale=osc,
              innerflowscale=isc, linepwr=lpw, dotpwr=dpw)
