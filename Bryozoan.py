# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:50:19 2016

@author: Michelangelo

This script is meant to become a simple resistive-network model of a feedback
between flow and conduit size in a sheet-like bryozoan colony to see if the
rule that high-flow causes conduits to enlarge (and low-flow causes counduits
to shrink) can maintain stable chimneys.

METHODS FOR COLONY CLASS:
__init__ : Set up colony network
colonyplot: plot conductivities of internal edges (lines) and edges to outside
    (dots); overlays with plot of flow to outside (stars) and (arrows) flow
    between nodes
setouterconductivities : Modify conductivity of inner node-to-outside edges.
solvecolony : Solve for pressures, dC/dt, and flow within network (given
    incurrent flows into nodes)
IntegrateColony : Solves differential equations based on dC/dt set in __init__.
OutflowFraction : returns a measure of how much a node functions as a chimney
develop : Create new colony object with conductivities updated by integration
    of ODE

ATTRIBUTES OF COLONY OBJECTS:
 'Adjacency',
 'InFlow',
 'Incidence',
 'InnerConduits',
 'Laplacian',
 'OutflowConduits',
 'UpperAdjacency'
 'colinds',
 'dCdt_inner'
 'dCdt_outer'
 'm',
 'n',
 'rowinds',
 'xs',
 'ys',
 'ysjig'

Other functions
dCdt_default : Default function for calculating dConductivity/dt

DESIRED FEATURES:
1) Methods to do the following:
x1.1 Improve update colony conductivities based on flow. Now have functions to
    calculate dC/dt (based on user-defined function) and integrate the ODE, but
    seems slow.
x1.2 Punch a hole in the colony (locally modify the conductivities, and keep
    them fixed).
x1.3 Assess the pattern (e.g. it's stability, chimneyishness, or aspects of
    performance)
1.4 Grow the colony
1.5 To speed up search through parameter space, define a function to assess
    whether a parameter set is satisfactory based on dConductivity/dt at a
    specific initial condition.

2) Averaging over nearby edges (conduits) to mimic the effect of having
multiple flow paths (with correlated conductivity) associated with each zooid.
Could probably implement by multiplying 'S' in dCdt_default() by the sum of
edges sharing vertices with a given edge (will need to add incidence matrix as
an input: change __init__ and solvecolony too; should end up being something
like: Incidence*transpose(Incidence))

3) Asymmetry in flow response and aging (so zooids can respond differently to
increased vs decreased flow, and old zooids respond differently than young
ones)

4) Maybe use tuples in some places I used lists (because tuples immutable), or
use attributes with leading underscore (e.g. '_x') to limit accidental changes
of calculated attributes?

5) Add limits/interactions on outflow conductivity. If outfloc conductivity can
grow without bound, it is unrealistic (because their combined areas cannot be
greater than the colony area), and may mean that it is hard to have stable
conduits within the colony (because flow through one conduit should be easier
than flow through a series).

6) Minimum conductivity for conduits that's greater than zero to mimic the
features of bryoz. colonies more closely.

#2 would add realism to the model, and may be important for stability, but may
not be common to other similar systems; it is unknown if #3 would add realism
(though seems likely) but similar effects occur in other systems and could
enhance stability. #5 would add realism, and should be similar to any system
with similar geometry, but adds extra parameters.

"""
# The following two lines should reset IPython (clear variables and libraries;
# alternative: type %reset in console).
from IPython import get_ipython
get_ipython().magic('reset -sf')

import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import scipy.sparse as sparse  # Sparse matrix library
from matplotlib.collections import LineCollection
from scipy.sparse.linalg import bicgstab
from scipy.integrate import ode


def dCdt_default(Cs, dPs, params):
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
        params must contain keys 'w', 'yminusx', 'r', and 'b' for parameters
        (all numeric types).

    Returns
    -------
    tuple : length 2, first part is an array of values for dConductivity/dt,
        second part is an array of values for quantifier S. Both arrays are
        have 1 dimension, same length as input arrays.

    Justification for form of S & dC/dt:
    ------------------------------------
    shearStress*perimeter*length = pressureDrop*crosssectionArea

    h = parameter with units of length (radius, width, height...)
        describing conduit width

    perimeter ~ A*h^x, 0 ≤ x ≤ 1 (A is constant of proportionality;
        x = 0 if h is separation between infinite parallel plates;
        x = 1 if h is radius or width of conduit that scales
        isotropically, e.g. radius of cylindrical pipe).
    crosssectionArea ~ K*h^y, 1 ≤ y ≤ 2 (y = 1 if h is separation between
        infinite parallel plates; y = 2 if h is radius or width of conduit
        that scales isotropically (e.g. radius of cylindrical pipe);
        BUT, for height of vertical parallel plates, area and perimeter
        both x and y increase in direct proporion to height: y = x = 1.
    Therefore, assuming conduit length is constant:
    shearStress ~ pressureDrop*a*h^(y-x) : 0 ≤ (y-x) ≤ 1 (with a = K/(A*length)).

    conductivity ~ d*h^w, 1 ≤ w ≤ 4 (w = 3 if h is separation of plates;
        w = 4 if h is radius of cylindrical pipe; w = 1 if h is height of
        parallel vertical plates, assuming their separation is much smaller
        than their height; d is a constant).

    Hence: h ~ (conductivity/d)^(1/w) and:
    shearStress ~ pressureDrop*(a/(d^z))*conductivity^z; z=(y-x)/w, b=a/(d^z)

    For the three cases above:
        separation between parallel plates: x = 0, y = 1, w = 3 : z = 1/3
        height of vertical parallel plates: x = 1, y = 1, w = 1 : z = 0
        radius of cylinder:                 x = 1, y = 2, w = 4 : z = 1/4

    Therefore, set S = b*(conductivity^z)*pressureDrop, 0 ≤ z ≤ 1/3
    As matrices (conductivity as diagonal matrix)
        S = b*abs(sum(conductivity[i,j]^z*sum(Incidence[j,k]*Pressures[k])
        
    Assuming dh/dt=r*(S-s0) (r is a constant), dC/dt = (d*h^(w-1))*dh/dt
    hence: dC/dt = d*(C/d)^((w-1)/w)*r*(S-s0) 
    dC/dt = d^((2w-1)/w) * C^((w-1)/w))*r*(S-s0) 
    One can choose parameters such that:
    dC/dt = r2*(C^((w-1)/w))*(S-1)
    """
    w = params.get('w')
    z = params.get('yminusx')/w
    # Added line to set floor on conductivities.
    Cs[Cs < 0] = 0
    S = abs(params.get('b')*(Cs**z)*dPs)
    dCdt = params.get('r') * (Cs**((w-1)/w)) * (S - 1)
    return dCdt, S


class Colony:
    """
    The Colony class represents the connections and arrangement of zooids
    in a sheet-like bryozoan colony.

    The colony geometry is a cylinder: the left and right sides connect to
    each other. Because of the hexagonal packing of lophophores (the
    tentacle crowns driving the flow and forming the conduits) it is easier
    to represent the colony spiraling around the cylinder (hence the slant
    in the plot of the x-y positions of the nodes). The lower side has
    closed boundary (or mirror symmetry). Currently, the upper boundary is
    also closed, but in future iterations it should be modified to become
    the growing edge, with special outflow conditions.

    Inner nodes (plotted as dots by ColonyPlot) represent the corners
    where three lophophores meet, so 6 nodes surround each lophophore.
    Hence, the inner edges (plotted as lines) are the open spaces under the
    canopy between the bases of the lophophores. Only inner nodes and edges
    are represented in the Incidence, Laplacian, and Adjacency matrices
    stored as Colony attributes. Their conductivities are represented in
    InnerConduits.

    Outflow conductivities represents gaps allowing flow to the outside
    (including chimneys). These edges (in OutflowConduits) connect the
    inner nodes to the outside. They are added to the network in
    SolveColony, but it is easier to treat them separately in other methods
    because they behave differently.
    """
    def __init__(self, nz=1, mz=1, InnerConductivity=1, OutflowConductivity=1,
                 Incurrents=-1, dCdt=dCdt_default,
                 dCdt_in_params={'yminusx': 0.5, 'b': 0, 'r': 0, 'w': 2},
                 dCdt_out_params={'yminusx': 1, 'b': 0, 'r': 0, 'w': 4}):
        """
        Create a new colony object given the following inputs:

        Parameters
        ----------
        nz : int
            number of zooid columns (run proximal-distal)
        mz : int
            number of zooid rows (run left-right)
        InnerConductivity : float
            Conductivity for edges (conduits) connecting nodes within colony
            (inner nodes).
        OutflowConductivity : float
            Conductivity between inner nodes and outside node.
        Incurrents : float
            Flow into colony (negative = inflow)
        Sfunc_... : Functions
            Calculate S, which measures match between conductivities & flow
            (using pressure differences). ...inner, ...outer are for conduits
            connecting inner-inner (or inner-growth zone), or inner-outer nodes
        """
        # Set up numbers of nodes.
        n = nz * 2  # 2 nodes added for every zooid from left-right;
        m = mz  # 1 node added for every zooid proximal-distal
        mn = np.arange(0, m * n)  # Number nodes. One node added on distal end
        # for every zooid row added.
        self.n = n
        self.m = m

        # Determine y and x coordinates.
        self.ys = mn // n  # Y position of nodes.
        self.xs = mn % n+self.ys  # X position of nodes
        self.ysjig = self.ys + 0.2 * (mn % 2)  # Y positions, shifted to
        # make hexagons for plotting.

        # Define indices for node-node connections in adjacency matrix.
        # Rowinds and colinds define arrays of indices for which internal nodes
        # connect to each other (in upper triangular matrix).
        # Each node from 0 to m*n-1 connects to next node in the network.
        rowinds = np.arange(0, m*n)
        colinds = np.arange(1, m*n+1)
        # delete edges connecting nodes at end of one row to beginning of next
        rowinds = np.delete(rowinds, np.arange(n-1, m*n, n))
        colinds = np.delete(colinds, np.arange(n-1, m*n, n))
        # add edges connecting beginning of row to end of row
        rowinds = np.concatenate((rowinds, np.arange(0, (m-1)*n+1, n)), axis=0)
        colinds = np.concatenate((colinds, np.arange(n-1, m*n, n)), axis=0)
        # add edges connecting every other node to node in row ahead.
        rowinds = np.concatenate((rowinds, np.arange(1, (m-1)*n, 2)), axis=0)
        colinds = np.concatenate((colinds, np.arange(n, m*n, 2)), axis=0)
        self.rowinds = rowinds
        self.colinds = colinds

        # Define conductivity among interior nodes. Fill in one value,
        # InnerConduits, for conductivities among nodes within colony.
        self.InnerConduits = np.array([InnerConductivity] * len(rowinds))

        # Combine node connection indices to form upper triangular part of
        # adjacency matrix for internal nodes
        UpperAdjacency = sparse.coo_matrix(
            ([1] * len(rowinds), (rowinds, colinds)), [m*n, m*n])
        self.UpperAdjacency = UpperAdjacency
        Adjacency = UpperAdjacency + UpperAdjacency.transpose()
        self.Adjacency = Adjacency.tocsr()

        # Create degree matrix (connections per node) for internal nodes
        Degree = sparse.diags(
            Adjacency.sum(axis=1).transpose().tolist()[0], 0)

        # Create Laplacian matrix of internal nodes
        self.Laplacian = (Adjacency - Degree).tocsr()

        # COO matrix sorts row indices, so convenient for finding sorted edges
        # to make incidence matrix for internal nodes.
        self.Incidence = (sparse.coo_matrix(([-1]*len(rowinds),
                          (np.arange(len(rowinds)), rowinds)),
                          [UpperAdjacency.nnz, n*m]) +
                          sparse.coo_matrix(([1]*len(colinds),
                                             (np.arange(len(colinds)),
                                              colinds)),
                          [len(rowinds), n*m]))

        # Define default conductivity for leakage from internal nodes to
        # outside node.
        self.OutflowConduits = np.array([OutflowConductivity]*(m*n))

        # Set default inflow magnitudes at each node
        self.InFlow = np.array([Incurrents]*(m*n))

        # Still need to add A) edges going out of colony

        # Set parameters for function for determining dConductivity/dt.
        self.dCdt_inner = lambda x, y: dCdt(x, y, dCdt_in_params)
        self.dCdt_outer = lambda x, y: dCdt(x, y, dCdt_out_params)

    def setouterconductivities(self, nodeinds, NewOuterConductivities):
        """
        Modify conductivity of edges connecting inner nodes (colony) to
        outside.

        Parameters
        ----------
        nodeinds : list
            List of indices of nodes to change (m*n>int>0)
        NewOuterConductivities : list
            List of new conductivities to apply (numeric & >0, same length as
            nodeinds)
        """
        if not (isinstance(NewOuterConductivities, list) and
                isinstance(nodeinds, list)):
            print('Arguments must be lists. No action taken.')
            return
        elif len(NewOuterConductivities) != len(nodeinds):
            print('List lengths must match. No action taken.')
            return
        elif (max(nodeinds) > (self.m*self.n)):
            print('At least one node index is >m*n. No action taken.')
            return
        elif (min(nodeinds) < 0):
            print('At least one node index is <0. No action taken.')
            return
        elif any(not isinstance(item, int) for item in nodeinds):
            print('At least one node index is not an int. No action taken.')
            return
        elif any(item < 0 and isinstance(item, (int, float))
                 for item in NewOuterConductivities):
            print('Conductivities must be floats or ints ≥0. No action taken.')
            return
        else:
            for k in range(len(nodeinds)):
                self.OutflowConduits[nodeinds[k]] = NewOuterConductivities[k]

    def solvecolony(self, calcpressures=True, calcflows=False, calcdCdt=False,
                    **kwargs):
        """
        Solve matrix equations for pressures at nodes, flows between nodes, and
        'shear-like' property (S) and dC/dt (a function of S).

        Because often want to pass different results of different steps in
        calculating pressures to other calculations, bring them all togeher
        here.

        First: Combines vectors of conductivities of internal edges
        (conduits) and edges connecting internal (colony) nodes to outside
        into one diagonal matrix. (conductivitymat = C)

        Second: Adds edges connecting internal nodes to outside to incidence
        matrix. (IncidenceFull = E)

        Third: Calculates pressures based on given inflow rates (from pumps)
        assuming volume conservation (Kirschoff's law). Vector of inflow rates,
        InFlow = q Pressure list = Pressures = p.
        Therefore, solve transpose(E)*C*E*p == q for p

        Fourth: Calculate flows (f) among all nodes (including outside node) as
        C*E*p = f

        Finally: Calculate dC/dt & 'S' (quantifier of flow vs conductivity).

        Parameters
        ----------
        self : colony object
        **kwargs : dictionary
            Can contain full conductivity array and full incidence matrix
            (as sparse matrices), plus pressure matrix. This allows passing
            approximate solutions f(from earlier calls) to future calls to this
            function.
            conductivityfull : ndarray with concatenation of inner-inner and
                inner-outer conductivities
            IncidenceFull : csr_matrix (sparse) concatenating incidence matrix
                for inner-inner and inner-outer nodes/edges.
            Pressures : ndarray containing pressures at each node
            S : ndarray containing 'shear-like' measure of fit between
                conductivity and flow
            Flows : numpy matrix of flows along each edge/conduit
            dCdt : ndarray of values of dC/dt
        calcpressures : boolean, default is True
            True : calculate pressures
        calcflows : boolean, default is False
            True : caclulate flows based on pressures
        calcdCdt : boolean, default is False
            True: caclulate dC/dt and S based on pressures

        Returns
        -------
        dictionary :
            dictionary stores conductivityfull (matrix of all conductivities,
            concatenating inner and outer), IncidenceFull (incidence matrix
            concatenating inner & outer edges), pressures, flows, S, & dC/dt
        """
        # Combine inner and outflow conduits into one diagonal conductivity
        # matrix.
        if (kwargs.get('conductivityfull') is None):
            conductivityfull = np.concatenate(
                                (self.InnerConduits, self.OutflowConduits))
        else:
            conductivityfull = kwargs.get('conductivityfull')

        # Add edges to outside to incidence matrix. Note that only add entry
        # for internal node (tail of edge) not outside, because including
        # including outer node makes matrix only solvable to an additive
        # constant.
        if (kwargs.get('IncidenceFull') is None):
            IncidenceFull = sparse.vstack((
                self.Incidence, sparse.diags(([-1]*(self.m*self.n)), 0).tocsr()
                ))
        else:
            IncidenceFull = kwargs.get('IncidenceFull')

        # Calculate pressures based on Kirchoff's current law. A few tests
        # indicate that the biconjugate gradient stabilized method (bicgstab)
        # is almost 100x faster than the direct method; with an initial
        # pressure solution, it may be 2x faster still (though my test for that
        # may be biased: I used the direct solution as the initial estimate, so
        # it was already right on the best value.
        if calcpressures:
            if (kwargs.get('Pressures') is None):
                Pressures = bicgstab(IncidenceFull.transpose() *
                                     sparse.diags(conductivityfull, 0) *
                                     IncidenceFull,
                                     np.asmatrix(self.InFlow).transpose())[0]
            else:
                Pressures = bicgstab(IncidenceFull.transpose() *
                                     sparse.diags(conductivityfull, 0) *
                                     IncidenceFull,
                                     np.asmatrix(self.InFlow).transpose(),
                                     x0=kwargs.get('Pressures'))[0]
        else:
            Pressures = kwargs['Pressures']

        networksols = {"Pressures": Pressures, "conductivityfull":
                       conductivityfull, "IncidenceFull": IncidenceFull}

        # Calculate flows based on pressure, conductivities, and connectivity
        if calcflows:
            networksols["Flows"] = sparse.diags(
                            conductivityfull, 0)*IncidenceFull*np.asmatrix(
                            Pressures).transpose()

        # Calculate derivatives of conductivity with time and match between
        # flow and conduit size ('S' ~ shear in Murray's Law) based on
        # pressure, conductivities, and connectivity
        # First checks that this calculation is requested.
        if calcdCdt:
            # Calculate array (1 by n*m array) of pressure differences (dP)
            dP = np.asarray(abs(IncidenceFull *
                                np.asmatrix(Pressures).transpose())).flatten()
            # Split dP into array for connected interior pairs, and array for
            # interior-outside pairs
            dPinner = dP[:self.InnerConduits.size]
            dPouter = dP[self.InnerConduits.size:]
            # Split conducitivity into arrays for inner and outflow conduits
            innerCs = networksols['conductivityfull'][:len(self.InnerConduits)]
            outerCs = networksols['conductivityfull'][len(self.InnerConduits):]
            dCdt_i, S_i = self.dCdt_inner(innerCs, dPinner)
            dCdt_o, S_o = self.dCdt_outer(outerCs, dPouter)

            networksols["S"] = np.concatenate((S_i, S_o))
            networksols["dCdt"] = np.concatenate((dCdt_i, dCdt_o))

        return networksols

    def colonyplot(self, addspy=True, linescale=1, dotscale=10,
                   outflowscale=10, innerflowscale=40, linepwr=1, dotpwr=1):
        """
        Create plots of colony object properties

        Parameters :
        ------------
         self : Colony object
         addspy: Boolean
             Whether to create optional plot of adjacency matrix
         linescale : numeric
             Multiply line widths (inner conductivities) by scalar.
         dotscale : numeric
             Multiply dot sizes (conductivity to outside) by scalar.
         outflowscale : numeric
             Multiplies symbol for outflow width
         innerflowscale : numeric
             length of inner flow arrows is divided by innerflowscale.
         linepwr : numeric
             allows control of whether line width is proporional to
             conductivity or conductivity raised to a power (e.g. to represent
             geometrical parameter)
         dotpwr : numeric
             allows control of whether dot area is proportional to
             conductivity or conductivity raised to a power.

        Plots produced: Plots circles for inner nodes (scaled by conductivity
        to outside node), lines for edges between inner nodes (width scaled to
        conductivity), a quiver plot for flows between inner nodes, and stars
        for flows to outside node (scaled by flow magnitude).
        """
        plt.figure()  # Create new figure.
        # Plot lines for edges among internal nodes; line width: conductivity
        # Convert coordinates of node-pairs to x-y coordinates of line segments
        segments = np.stack((np.vstack((self.xs[self.rowinds],
                                        self.xs[self.colinds])),
                             np.vstack((self.ysjig[self.rowinds],
                                        self.ysjig[self.colinds]
                                        )))).transpose()
        # Create matplotlib.collections.LineCollection object from segments,
        # with widths defined by conduit conductivity
        edges = LineCollection(segments, zorder=1,
                               linewidths=np.dot(linescale,
                                                 self.InnerConduits**linepwr))
        # Plot segments.
        plt.gca().add_collection(edges)
        # Only included these two lines setting xlim & ylim for ease if want to
        # plot just this part; not necessary if scatter plot defines x-y axes.
        plt.xlim(-0.5, self.xs.max() + 0.5)
        plt.ylim(-0.5, self.ysjig.max() + 0.5)

        # Make scatter plot of outflow conduit conductivities (conductivities
        # between internal nodes and outside.)
        plt.scatter(self.xs, self.ysjig, c='c',
                    s=np.dot(dotscale, self.OutflowConduits**dotpwr), zorder=2)

        # Solve for flows in network. solveflow returns flows; convert flow
        # matrix to array.
        Flows = np.array(self.solvecolony(calcflows=True).get("Flows"))
        # Separate inner and outer flows
        OuterFlows = Flows[-len(self.OutflowConduits):]
        InnerFlows = Flows[:len(self.InnerConduits)].flatten()
        # Plot flow from nodes to outside
        plt.scatter(self.xs, self.ysjig, s=(outflowscale*OuterFlows).tolist(),
                    c='r', marker='*', zorder=3)
        # Plot flows between inner nodes. First get orientation vector (not a
        # unit vector) and its magnitude to use to determine x, y components of
        # flow vectors.
        Orientation_Vect = np.vstack((self.xs[self.colinds] -
                                      self.xs[self.rowinds],
                                      self.ysjig[self.colinds] -
                                      self.ysjig[self.rowinds]))
        Mag_Orientation_Vect = sum(Orientation_Vect**2)**(0.5)
        plt.quiver((self.xs[self.rowinds] + self.xs[self.colinds])/2,
                   (self.ysjig[self.rowinds] + self.ysjig[self.colinds])/2,
                   InnerFlows * Orientation_Vect[0, :]/Mag_Orientation_Vect,
                   InnerFlows * Orientation_Vect[1, :]/Mag_Orientation_Vect,
                   color='r', pivot='mid', scale=innerflowscale, zorder=4)

        # Optional plot of adjacency matrix.
        if addspy:
            plt.figure()
            plt.spy(self.Adjacency)

    def IntegrateColony(self, tmax=1):
        """
        ODE integration of conductivity over time as defined by self.dCdt
        odeint() seemed slow and error prone; therefore switched to ode() with
        RungaKutta method (dopri5).

        This variant simply sets a floor of zero on conductivities.

        Parameters
        ----------
        self : colony object
        tmax : float or int
            time to integrate ODE to
        params : dict
            dictionary of parameters for integrator

        Returns
        -------
        List : each element is a list [t, C] with t the time point of the
            integration step, and C a numeric numpy.ndarray of conductivities
            at that step (C is flattened with dimensions: 1 * #edges;
            C[0:self.InnerConduits.size] are innerconduits;
            C[self.Innerconduits.size:] are outflow conduits).
        """
        params = self.solvecolony(calcdCdt=False, calcflows=False)
        C0 = params.get('conductivityfull')

        def dCdt_simpleinputs(t, C0):
            """
            Converts dCdt into a form that ODE solver can handle. Uses 'params'
            and 'self' defined in enclosing scope.

            self.solvecolony() takes pressures saved in 'params' as a starting
            guess when solving for pressure. Updating the starting guess with
            each call to dCdt might speed up solution unless the ODE solver
            overshoots too much.

            Parameters
            ----------
            t : n/a
                Dummy variable required by numpy.integrate.ode syntax
            C0 : Numpy.ndarray
                Full list of conductivity for all edges (inner + outflow)

            Returns
            -------
            numpy.ndarray of derivatives of conductivity with time
            """
            C0 = np.maximum(C0, 0)
            dCdt_vals = self.solvecolony(calcdCdt=True, calcflows=False,
                                         Pressures=params.get('Pressures'),
                                         IncidenceFull=params.get(
                                                             'IncidenceFull'),
                                         conductivityfull=C0).get('dCdt')
            return dCdt_vals

        y = ode(dCdt_simpleinputs)
        # Tends to take first step too big if tmax is set high, so set initial
        # step size to try to prevent values from going below 0 on first step.
        dCdt0 = dCdt_simpleinputs(0, C0)
        problemvals = dCdt0 < 0
        dt0 = 0.5 * np.min(abs(
                         C0[problemvals]/dCdt0[problemvals]))
        y.set_integrator('dopri5', first_step=dt0, nsteps=2000)
        sol = []

        def solout(tcurrent, ytcurrent):
            # ytcurrent.copy() prevents ytcurrent arrays in sol from being
            # duplicates and getting set to zero or garbage at the end.
            sol.append((tcurrent, ytcurrent.copy()))

        y.set_solout(solout)
        y.set_initial_value(y=C0, t=0)
        # yfinal = y.integrate(tmax)
        y.integrate(tmax)

        return sol

    def OutflowFraction(self, nodeind=None):
        """
        An index of whether the colony functions as a chimney system

        Parameters :
        ------------
        self : colony object
        nodeind : int
            test chimney function of outflow conduit at nodeind; if nodeind is
            None (i.e. unspecified), uses outflow conduit with maximum flow.

        Returns :
        float, ratio of outflow at nodeind to total outflow
        """
        # Solve for flows in network. solveflow returns flows
        Outflows = (self.solvecolony(calcflows=True).get("Flows")
                    )[-len(self.OutflowConduits):]
        if nodeind is None:
            ChimOutflow = max(Outflows)
        else:
            ChimOutflow = Outflows[nodeind]

        return self.OutflowConduits.size * (ChimOutflow/np.sum(Outflows))[0, 0]

    def develop(self, tmax=1):
        """
        Create new colony object with conductivities updated by integration
        of ODE

        Parameters :
        ------------
        tmax : float
            Time to integrate over

        Returns :
        ---------
        newcolony : colony object with updated conductivities
        """
        ontogeny = self.IntegrateColony(tmax)
        newcolony = copy.deepcopy(self)
        newcolony.InnerConduits = np.copy(ontogeny[-1][1]
                                          )[0:len(self.InnerConduits)]
        newcolony.OutflowConduits = np.copy(ontogeny[-1][1]
                                            )[len(self.InnerConduits):]
        return newcolony

# For fast search of parameter space via one step differentiation, fastest to
# much faster to add if statement that calculate pressures from answer.

# Demonstration. Example of how these functions work.
if __name__ == '__main__':
    demo = input('To run demo type: y')
    if demo == 'y':
        # Create colony object.
        c1 = Colony(nz=6, mz=7, OutflowConductivity=0.01, dCdt=dCdt_default,
                    dCdt_in_params={'yminusx': 1, 'b': 1, 'r': 1, 'w': 3},
                    dCdt_out_params={'yminusx': 1, 'b': 0.1, 'r': 1, 'w': 3})
        # Set a central outflow conduit (edge) to have higher conductivity
        c1.setouterconductivities([41], [0.02])
        # Create plot
        c1.colonyplot(False, linescale=0.2, dotscale=80, outflowscale=20,
                      innerflowscale=80)
        # Solve dif. eqs. for c1 and put result in c2.
        t = time.time()
        c2 = c1.develop(3)
        print(time.time()-t)
        c2.colonyplot(False, linescale=0.2, dotscale=80, outflowscale=20,
                      innerflowscale=80)

        # Perturb colony c2 by opening up outflow in lower left, then plot.
        c2.setouterconductivities([0], [0.2])
        c2.colonyplot(False, linescale=0.2, dotscale=80, outflowscale=20,
                      innerflowscale=80)

        # Solve dif. eqs. for c2 after perturbation, put in c3 & plot.
        t = time.time()
        c3 = c2.develop(3)
        print(time.time()-t)
        c3.colonyplot(False, linescale=0.2, dotscale=80, outflowscale=20,
                      innerflowscale=80)

# MAY BE A PROBLEM USING SOLVER IF VALUES EVER GO NEGATIVE...PERHAPS IT WOULD
# WORK BETTER IF REFRAMED IN Ln(Conductivity) SO GOING NEGATIVE WOULDN'T CAUSE
# PROBLEMS? OR NEED SPECIAL CASE FOR NEGATIVE VALUES (JUST SET dC/dt = 0 for
# C=0)? LOG CONDUCTIVITY IS APPEALING BUT CAN'T SET ANY CONDUCTIVITIES TO 0)
# Tried using Log(conductivity) in solver. Seems to work a bit faster (for
# some circumstances), but still unreliable and slow.

# Now sets floor of conductivities to 0 in IntegrateColony and dCdt_default.
# Also set imaginary components to zeros?

#

"""
21Oct2016Calculation is very slow (72s), but the following may give a stable
chimney:
OutflowConductivity=0.01
dCdt_in_params={'yminusx': 1, 'b': 1, 'r': 10, 'w': 3}
dCdt_out_params={'yminusx': 1, 'b': 0.1, 'r': 10, 'w': 3}
And are somewhat reasonable (assuming small pores initially; b_out<<b_in means
threshold shear stress is higher for outflow conduits than inner conduits;
choice of w and yminusx for h = separation of parallel plates and h = radius of
pore at low Re, respectively.
"""

"""
Quantify chimneyishness for opening i as V = qi/sum(qj) for j within a radius
(2 edges?) of i: V would be a functional measure that should reflect structural
properties.

To speed up search of parameter space, estimate dV/dt by estimating
delta(C) to calculate delta(V)
"""
"""
02Nov2016: Found the solver fails when choose parameters (e.g. long time or
large rates) that cause the solver to go to values below zero based on its
automatic choice of initial step. I adjusted this to set it so that the inital
step size is low enough that it won't reach negative values in one step.
"""

"""12Nov2016: Something changes when calls SolveColony; it seems to need to
be called before calling IntegrateColony to get sensible results, even though
it is called as a first step in IntegrateColony...

13Nov2016: It turns out that one needs to use X.copy() when appending arrays
to sol. Otherwise a) they all are the same (usual trickyness of mutable values)
and b) they go to zero after it clears unless yfinal is saved.
"""
