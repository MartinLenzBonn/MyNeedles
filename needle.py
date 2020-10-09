from dolfin import *
from fenics import *
from mshr import *
from dolfin_adjoint import *
from pyadjoint.overloaded_type import create_overloaded_object
import numpy

parameters ["form_compiler"]["cpp_optimize"] = True
parameters ["form_compiler"]["representation"] = "uflacs"

Ll = 2; Ln = 4.5; Lr = 7.5
theta = 0.22
a1=11.56; a2=-17.44; a3=10.04; a4=-9.38
delta = 0.1

# Create domain with subdomains
domain = Rectangle (Point (-Ll, -0.5), Point (Ln+Lr, 0.5)) 
domain.set_subdomain (1, Rectangle (Point (-Ll, -0.5), Point (0, 0.5)))
domain.set_subdomain (2, Rectangle (Point (0, -0.5), Point (Ln, 0.5)))
domain.set_subdomain (3, Polygon ([Point (0, 0), Point (Ln, -0.5*theta), Point (Ln, 0.5*theta)]))
domain.set_subdomain (4, Polygon ([Point (Ln, -0.5*theta), Point (Ln+Lr, -0.5*theta), Point (Ln+Lr, 0.5*theta), Point (Ln, 0.5*theta)]))
with Timer ("mesh"): mesh = create_overloaded_object (generate_mesh (domain, 126)) # 126 253 505 1011 2022

class PeriodicBoundary (SubDomain):
    # bottom boundary is target domain
    def inside (self, x, on_boundary): return bool (near (x[1], -0.5) and on_boundary)
    # Map top boundary to bottom boundary
    def map (self, x, y): y[0] = x[0]; y[1] = x[1]-1.0

U = FunctionSpace (mesh, "CG", 1)
V = VectorFunctionSpace (mesh, "CG", 1, constrained_domain=PeriodicBoundary())
Vnp = VectorFunctionSpace (mesh, "CG", 1)
W = TensorFunctionSpace (mesh, "DG", 0)
X = FunctionSpace (mesh, "DG", 0)

x = SpatialCoordinate (mesh)
    
sudom = MeshFunction ('size_t', mesh, 2, mesh.domains())
sudom_arr = numpy.asarray (sudom.array(), dtype=numpy.int)
dm = X.dofmap()
for cell in cells (mesh): sudom_arr [dm.cell_dofs (cell.index())] = sudom [cell]

def sudom_fct (sudom_arr, vals, fctspace):
    f = Function (fctspace)
    f.vector()[:] = numpy.choose (sudom_arr, vals)
    return f

chi_a = sudom_fct (sudom_arr, [0,1,0,1,1], X)
chi_b = sudom_fct (sudom_arr, [1,0,1,0,0], X)
chi_l = sudom_fct (sudom_arr, [0,1,0,0,0], X)
chi_n = sudom_fct (sudom_arr, [0,0,1,1,0], X)
chi_r = sudom_fct (sudom_arr, [1,0,0,0,1], X)

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
alpha = Constant(0.0)

def domain_transformation (Ln, Lr, alpha, x, chi_l, cbhi_n, chi_r):
    L = Ln + alpha
    nfac=L/Ln
    rfac=(Ln+Lr-L)/Lr
    roff=L-rfac*Ln
    trf0 = chi_l*x[0] + chi_n*nfac*x[0] + chi_r*(rfac*x[0]+roff)
    trf1 = x[1]
    trf = as_vector((trf0,trf1))
    return trf
    
trf = domain_transformation (Ln, Lr, alpha, x, chi_l, chi_n, chi_r)    
T = grad (trf)

# eigenstrain
GA = Constant (((1,delta), (0,1)))
GB = Constant (((1,-delta), (0,1)))
G = chi_a*GA + chi_b*GB

# Kinematics
F = ( Identity(2) + grad(u)* inv(T) ) * inv(G)
C = F.T*F

# Stored strain energy density 
Edens = (a1*(tr(C))**2 + a2*det(C) - a3*ln(det(C)) + a4*(C[0,0]**2+C[1,1]**2) - (4*a1+a2+2*a4))*abs(det(T))

# Linear case
#d11 = 200 - (130**2)/200
#d12 = 130 - (130**2)/200
#d44 = 110
#G = (G+G.T)/2
#eps = (grad(u)*inv(T)+(grad(u)*inv(T)).T)/2 - G
#Edens = (1/2*d11*(eps[0,0]**2+eps[1,1]**2) + d12*eps[0,0]*eps[1,1] + 2*d44*eps[0,1]**2) * abs(det(T))

# Total potential energy and derivatives
E = Edens*dx
F = derivative (E, u, v)
J = derivative (F, u, du)

def eval_cb(j, m):
  print ("E = %10.8f   L = %f" % (j, Ln+float(m)))

with Timer ("solve"): 
    solve (F == 0, u, bcs, J=J, solver_parameters={"newton_solver":{"linear_solver":"mumps", "relative_tolerance":1e-12}})
    startE = assemble(E)
# mumps umfpack superlu petsc

ca = Control (alpha)
Ehat = ReducedFunctional (assemble (E), ca, eval_cb_post = eval_cb)

with Timer ("solve"): alpha = float(minimize (Ehat, method="L-BFGS-B", tol=1e-10, options={'disp': True}))
# L-BFGS-B Newton-CG BFGS
optL = Ln + alpha
with Timer ("solve"): optE = Ehat (alpha)

# Reconstruct transformation
trf = domain_transformation (Ln, Lr, alpha, x, chi_l, chi_n, chi_r)    

# Save solution
with Timer ("save"): 	
    fileResults = XDMFFile ("output.xdmf"); fileResults.parameters ["flush_output"] = True; fileResults.parameters ["functions_share_mesh"] = True
    trafo = Function (Vnp, name='Transformation'); trafo.assign (project (trf-x, Vnp)); fileResults.write (trafo, 0)
    disp = Function (Vnp, name='Displacement'); disp.assign (project (u,Vnp)); fileResults.write (disp, 0)
    disp = Function (Vnp, name='Trf*Disp'); disp.assign (project (trf-x+u,Vnp)); fileResults.write (disp, 0)
    strain = Function (W, name='Strain'); strain.assign (project (grad(disp), W)); fileResults.write (strain, 0)
    dete = Function (X, name='Determinant'); dete.assign (project (det(grad(disp)), X)); fileResults.write (dete, 0)
    need = Function (X, name='Chi_Needle'); need.assign (chi_a); fileResults.write (need, 0)
    
# Timings
print ("Number of nodes: ", V.dim())
print ("Mesh fineness:   ", mesh.hmax())
print ("Meshing time:    ", timing ("mesh", TimingClear.keep)[1])
print ("Solving time:    ", timing ("solve", TimingClear.keep)[1])
print ("Saving time:     ", timing ("save", TimingClear.keep)[1])
print ("Starting Energy: ", startE)
print ("Optimal Energy:  ", optE)
print ("Optimal Length:  ", optL)
# list_timings(TimingClear.keep, [TimingType.wall])