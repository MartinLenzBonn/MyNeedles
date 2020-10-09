import numpy
import scipy
import scipy.optimize

from dolfin import *
from fenics import *
from dolfin_adjoint import *

from mpi4py import MPI
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
starttime = MPI.Wtime()

parameters ["form_compiler"]["cpp_optimize"] = True
parameters ["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 1
gridfilename = "grid"

# info (NonlinearVariationalSolver.default_parameters (), True); exit()
# nonlin_solve_params = {"nonlinear_solver":"snes", "snes_solver":{"method":"newtontr", "linear_solver":"mumps", "maximum_iterations":100, "relative_tolerance":1e-12}}
nonlin_solve_params = {"nonlinear_solver":"newton", "newton_solver":{"linear_solver":"mumps", "maximum_iterations":100, "relative_tolerance":1e-12}}

# Read grid
mesh = Mesh()
gridfilename = "%s.h5" % (gridfilename)
gridfile = HDF5File (comm, gridfilename, "r")
gridfile.read (mesh, "/mesh", False)

# Read subdomain functions
X = FunctionSpace (mesh, "DG", 0)
chi_a = Function (X); gridfile.read (chi_a, "/chi_a")
chi_b = Function (X); gridfile.read (chi_b, "/chi_b")
chi_l = Function (X); gridfile.read (chi_l, "/chi_l")
chi_n = Function (X); gridfile.read (chi_n, "/chi_n")
chi_r = Function (X); gridfile.read (chi_r, "/chi_r")

hmax = numpy.array (0.0)
comm.Reduce (numpy.array (mesh.hmax()), hmax, op=MPI.MAX, root=0)

comm.barrier()
if rank == 0: print ("********** Read mesh with h = %12.10f from %s." % (hmax, gridfilename), flush=True)


# Recompute geometry parameters
Ll = assemble (chi_l * dx)
Ln = assemble (chi_n * dx)
Lr = assemble (chi_r * dx)
theta = assemble (chi_r * chi_a * dx) / Lr

a1=11.56; a2=-17.44; a3=10.04; a4=-9.38
delta = 0.1

comm.barrier()
if rank == 0: 
    print ("********** L = %f + %f + %f, H = 1, theta = %f" % (Ll, Ln, Lr, theta), flush=True)
    print ("********** a = (%f, %f, %f, %f), delta = %f" % (a1, a2, a4, a4, delta), flush=True)

class PeriodicBoundary (SubDomain):
    # bottom boundary is target domain
    def inside (self, x, on_boundary): return bool (near (x[1], -0.5) and on_boundary)
    # Map top boundary to bottom boundary
    def map (self, x, y): y[0] = x[0]; y[1] = x[1]-1.0

U = FunctionSpace (mesh, "CG", 1)
V = VectorFunctionSpace (mesh, "CG", 1, constrained_domain=PeriodicBoundary())
Vnp = VectorFunctionSpace (mesh, "CG", 1)

class DirichletBoundary (SubDomain):
    def inside (self, x, on_boundary): return bool (near (x[0], 0) and near (x[1], 0))

zero = Constant ((0,0))
tip = DirichletBoundary()
dbc = DirichletBC (V, zero, tip,method='pointwise')
bcs = [dbc]

dx = Measure ('dx', domain=mesh)

du = TrialFunction (V)
v  = TestFunction (V)
u  = Function (V)

# Domain transformation
L = Constant (Ln)
at = Constant (0)
ab = Constant (0)
Delta = Constant (0)

S1 = Constant(((1,-delta*theta/sqrt(1+delta**2+theta**2)),(0,1/sqrt(1+delta**2+theta**2))))
S2 = Constant(((0,(2*delta*theta-delta)/sqrt(1+delta**2+theta**2)),(0,0)))

x = SpatialCoordinate (mesh)
chi_abo = project (conditional(gt(x[1],0),1,0), X)
chi_belo = project (conditional(lt(x[1],0),1,0), X)

def domain_transformation (Ln, Lr, theta, x, chi_l, cbhi_n, chi_r, L, at, ab, Delta):
    nfac=L/Ln
    rfac=(Ln+Lr-L)/Lr
    roff=L-rfac*Ln
    g_t = -at*x[0]**2 + ((Delta+theta/2)/Ln+at*Ln)*x[0] 
    g_b = ab*x[0]**2 + ((Delta-theta/2)/Ln-ab*Ln)*x[0] 
    lam_need= x[1]*Ln/(x[0]*theta) + 0.5
    lam_abo = (1-2*x[1])/(1-x[0]*theta/Ln)
    lam_belo = (1+2*x[1])/(1-x[0]*theta/Ln)
    lam_abo_r = (1-2*x[1])/(1-theta)
    lam_belo_r = (1+2*x[1])/(1-theta) 
    trf0 = chi_l*x[0] + chi_n*nfac*x[0] + chi_r*(rfac*x[0]+roff)
    trf1 = x[1] + chi_a*chi_n*(lam_need*(g_t-x[1])+(1-lam_need)*(g_b-x[1])) \
                    + chi_b*chi_n*chi_abo*(lam_abo*g_t+(1-lam_abo)*0.5-x[1]) + chi_b*chi_n*chi_belo*(lam_belo*g_b-(1-lam_belo)*0.5-x[1]) \
                    + chi_a*chi_r*Delta + chi_b*chi_r*chi_abo*lam_abo_r*Delta + chi_b*chi_r*chi_belo*lam_belo_r*Delta
    trf = S1*as_vector((trf0,trf1))
    return trf
    
trf = domain_transformation (Ln, Lr, theta, x, chi_l, chi_n, chi_r, L, at, ab, Delta)    
T = grad (trf)

# eigenstrain
GA = Constant (((1,delta), (0,1)))
GB = Constant (((1,-delta), (0,1)))
G = chi_a*GA + chi_b*GB

def energy_density (u, T, G, a1, a2, a3, a4):
    F = ( Identity(2) + S2 + grad(u)* inv(T) ) * inv(G)
    C = F.T*F
    return (a1*(tr(C))**2 + a2*det(C) - a3*ln(det(C)) + a4*(C[0,0]**2+C[1,1]**2) - (4*a1+a2+2*a4))*abs(det(T))

# Linear case
#d11 = 200 - (130**2)/200
#d12 = 130 - (130**2)/200
#d44 = 110
#G = (G+G.T)/2
#eps = (grad(u)*inv(T)+(grad(u)*inv(T)).T)/2 - G
#Edens = (1/2*d11*(eps[0,0]**2+eps[1,1]**2) + d12*eps[0,0]*eps[1,1] + 2*d44*eps[0,1]**2) * abs(det(T))

# Total potential energy and derivatives
Edens = energy_density (u, T, G, a1, a2, a3, a4)
E = Edens*dx
F = derivative (E, u, v)
J = derivative (F, u, du)

def eval_cb_pre(m):
    if rank == 0: print ("** Evaluating at        L = %10.8f     at = %10.8f     ab = %10.8f     Delta = %10.8f" % (float(m[0]), float(m[1]), float(m[2]), float(m[3])), flush=True)

def eval_cb_post(j, m):
    if rank == 0: print ("** Evaluated energy     E = %10.8f" % (j), flush=True)

def deriv_cb_post(j, dj, m):
    if rank == 0: print ("** Evaluated gradient d L = %10.8f   d at = %10.8f   d ab = %10.8f   d Delta = %10.8f" % (float(dj[0]), float(dj[1]), float(dj[2]), float(dj[3])), flush=True)

solve (F == 0, u, bcs, J=J, solver_parameters=nonlin_solve_params)
startE = assemble(E)

ctrl = [Control (L), Control (at), Control (ab), Control (Delta)]
Ehat = ReducedFunctional (assemble (E), ctrl, eval_cb_post = eval_cb_post, eval_cb_pre = eval_cb_pre, derivative_cb_post = deriv_cb_post)

# print_optimization_methods(); scipy.optimize.show_options ('minimize'); exit()
# optctrl = minimize (Ehat, method="L-BFGS-B", bounds=[[Ln*0.5,0,0,-0.1],[Ln+Lr*0.5,0.01,0.01,0.1]], tol=1e-10, options={'disp': True})
optctrl = minimize (Ehat, method="TNC", bounds=[[Ln*0.5,0,0,-0.05],[Ln+Lr*0.5,0.01,0.01,0.05]], tol=1e-10, options={'maxCGit':10, 'eta':0.5, 'scale':[10,0.01,0.01,0.1]})
[L, at, ab, Delta] = optctrl

# Reconstruct transformation and recompute solution
trf = domain_transformation (Ln, Lr, theta, x, chi_l, chi_n, chi_r, L, at, ab, Delta)    
T = grad (trf)

Edens = energy_density (u, T, G, a1, a2, a3, a4)
E = Edens*dx
F = derivative (E, u, v)
J = derivative (F, u, du)

solve (F == 0, u, bcs, J=J, solver_parameters=nonlin_solve_params)
optE = assemble (E)

# Save solution
fileResults = XDMFFile ("output.xdmf"); fileResults.parameters ["flush_output"] = True; fileResults.parameters ["functions_share_mesh"] = True
trafo = Function (Vnp, name='Transformation'); trafo.assign (project (trf-x, Vnp)); fileResults.write (trafo, 0)
disp = Function (Vnp, name='Displacement'); disp.assign (project (u,Vnp)); fileResults.write (disp, 0)
tdisp = Function (Vnp, name='Trf*Disp'); tdisp.assign (project (trf-x+u,Vnp)); fileResults.write (tdisp, 0)
edens = Function (U, name='Energy'); edens.assign (project (Edens, U)); fileResults.write (edens, 0)
need = Function (X, name='Chi_Needle'); need.assign (chi_a); fileResults.write (need, 0)
    
# Timings
comm.barrier()
if rank == 0:
    sys.stdout.flush()
    print ("********** L = %f + %f + %f, H = 1, theta = %f" % (Ll, Ln, Lr, theta))
    print ("********** a = (%f, %f, %f, %f), delta = %f" % (a1, a2, a4, a4, delta))
    print ("********** %12s    %12s  (np)    %12s    %12s    %12s    %12s    %12s    %12s" % ("mesh h", "runtime", "start E", "opt E", "opt Ln", "opt a^t", "opt a^b", "opt Delta"))
    print ("********** %12.10f    %12.3f  (%2i)    %12.10f    %12.10f    %12.10f    %12.10f    %12.10f    %12.10f" % (hmax, MPI.Wtime()-starttime, size, startE, optE, L, at, ab, Delta));
    # list_timings(TimingClear.keep, [TimingType.wall])