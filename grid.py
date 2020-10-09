from dolfin import *
from fenics import *
from mshr import *
import numpy

from mpi4py import MPI
comm=MPI.COMM_WORLD

Ll = 2; Ln = 7.0; Lr = 5.0
theta = 0.22
resolution = 300
gridfilename = "grid"

# Create domain with subdomains
domain = Rectangle (Point (-Ll, -0.5), Point (Ln+Lr, 0.5)) 
domain.set_subdomain (1, Rectangle (Point (-Ll, -0.5), Point (0, 0.5)))
domain.set_subdomain (2, Rectangle (Point (0, -0.5), Point (Ln, 0.5)))
domain.set_subdomain (3, Polygon ([Point (0, 0), Point (Ln, -0.5*theta), Point (Ln, 0.5*theta)]))
domain.set_subdomain (4, Polygon ([Point (Ln, -0.5*theta), Point (Ln+Lr, -0.5*theta), Point (Ln+Lr, 0.5*theta), Point (Ln, 0.5*theta)]))
mesh = generate_mesh (domain, resolution)

# Create characteristic functions
X = FunctionSpace (mesh, "DG", 0)
dm = X.dofmap()
sudom = MeshFunction ('size_t', mesh, 2, mesh.domains())
sudom_arr = numpy.asarray (sudom.array(), dtype=numpy.int)
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

# Save mesh and subdomain functions
gridfilename = "%s.h5" % (gridfilename)
gridfile = HDF5File (comm, gridfilename, "w")
gridfile.write (mesh, "/mesh")
gridfile.write (chi_a, "/chi_a")
gridfile.write (chi_b, "/chi_b")
gridfile.write (chi_l, "/chi_l")
gridfile.write (chi_n, "/chi_n")
gridfile.write (chi_r, "/chi_r")

print ("Generated mesh with %i nodes and h = %f, written to %s." % (mesh.num_vertices(), mesh.hmax(), gridfilename))

