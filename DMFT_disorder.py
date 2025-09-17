from triqs.gf import *
from triqs.operators import *
from h5 import *
from triqs.utility import mpi
from nrgljubljana_interface import Solver, MeshReFreqPts, hilbert_transform_refreq
import math, os, warnings
import numpy as np
from scipy import interpolate, integrate, special, optimize
from collections import OrderedDict
from scipy.integrate import simpson
 
# Parameters
model = "SIAM"        
dos = "Bethe"             # "Bethe" for a Bethe lattice (semicircular) DOS or a file name for reading tabulated DOS data from a file;
                          # format of the DOS file: lines with "eps rho(eps)" pairs; arbitrary bandwidth, but should be within the NRG mesh
Bethe_unit = 1.0          # half-bandwidth in the Bethe lattice DOS for dos="Bethe"; 1.0 for D=1 units, 2.0 for t=1 units
W = 0.5                   # Disorder strength
N = 6                     # Number of disorder values
U = 1.75             
T = 1.0e-1          
B = None                  # magnetic field for spin-polarized calculation with U(1) symmetry;
                          # None = non-spin-polarized calculation with SU(2) symmetry
omega = None              # Holstein phonon frequency
g1 = None                 # electron-phonon coupling strength
n1 = None                 # shift n1 in the Holstein model definition
verbose = True            # show info messages during the iteration
verbose_nrg = False       # show detailed output from the NRG solver
store_steps = True        # store intermediate results to files (one subdirectory per iteration)
min_iter = 1       
max_iter = 1000   
eps_prev = 1e-8           # convergence criterium: integrated diff between two consecutive local spectral functions
eps_loc_imp = 1e-7        # convergence criterium: integrated diff between local and impurity spectral function
mixing_method = "broyden" # "linear" or "broyden" mixing; the quantity being mixed is the hybridisation function
alpha = 0.5               # mixing parameter from both linear and Broyden mixing
mesh_max = 10.0           # logarithmic mesh: maximum frequency (should be large enough to contain the full spectrum)
mesh_min = 1e-5           # logarithmic mesh: minimum frequency (should be smaller than the temperature T)
mesh_ratio = 1.01         # logarithmic mesh: common ratio of the mesh (1.01 is usually a good choice)
Delta_min = 1e-5          # minimal value for the hybridisation function; if too large, it produces incorrect spectral distribution,
normalize_to_one = True   
solution_filename = "solution.h5"
checkpoint_filename = "checkpoint.h5"

Sigm = np.loadtxt('Sigma_imp.dat')

mu=U/2
V=0.44
tp=1.033

observables = ["n_d", "n_d^2"]
if B is not None:
  observables.extend(["SZd"])
if "Holstein" in model:
  observables.extend(["nph", "displ", "displ^2"])

# Run-time global variables
itern = 0    
verbose = verbose and mpi.is_master_node()         # output is only produced by the master node
store_steps = store_steps and mpi.is_master_node() # output is only produced by the master node
symtype = ("QS" if B is None else "QSZ")

S = Solver(model=model, symtype=symtype, mesh_max=mesh_max, mesh_min=mesh_min, mesh_ratio=mesh_ratio)
S.set_verbosity(verbose_nrg)

newG = lambda : S.G_w.copy()    
nr_blocks = lambda bgf : len([bl for bl in bgf.indices]) # Returns the number of blocks in a BlockGf object
block_size = lambda bl : len(S.G_w[bl].indices[0])       # Matrix size of Green's functions in block 'bl'
identity = lambda bl : np.identity(block_size(bl))       # Returns the identity matrix in block 'bl'


if (dos == "Bethe"):
  ht1 = lambda z: 2*(z-1j*np.sign(z.imag)*np.sqrt(1-z**2)) 
  global ht0
  ht0 = lambda z: ht1(z/Bethe_unit)
else:
  table = np.loadtxt(dos)
  global dosA
  dosA = Gf(mesh=MeshReFreqPts(table[:,0]), target_shape=[])
  for i, w in enumerate(dosA.mesh):
      dosA[w] = np.array([[ table[i,1] ]])
  ht0 = lambda z: hilbert_transform_refreq(dosA, z)

EPS = 1e-20 
ht = lambda z: ht0(z.real+1j*(z.imag if z.imag>0.0 else EPS))
ht2 = lambda z: ht0(z.real + 1j * np.where(z.imag > 0.0, z.imag, EPS))

index_range = lambda G : range(len(G.indices[0]))

def generate_disorder(N, W):
    np.random.seed(42)
    pos_dis = np.random.uniform(0, W/2.0, int(N/2))
    return np.concatenate((-np.flip(pos_dis), pos_dis))

class Converged(Exception):
  def __init__(self, message):
      self.message = message

class FailedToConverge(Exception):
  def __init__(self, message):
      self.message = message

def new_calculation():
  if verbose: print("Starting from scratch\n")
  Sigma = newG()
  for bl in Sigma.indices:
    for ss,w in enumerate(Sigma.mesh):
        Sigma[bl][w] = Sigm[ss,1]+1j*Sigm[ss,2]
        #Sigma[bl][w] = U/2.0 
  return Sigma

# Calculate a GF from hybridisation and self-energy
def calc_G(Delta, Sigma, mu):
  G = newG()
  for bl in G.indices:
    for w in G.mesh:
      G[bl][w] = np.linalg.inv( (w + mu)*identity(bl) - Delta[bl][w] - Sigma[bl][w] ) # !!!
      #G[bl][w] = np.linalg.inv( (w + 1j*1e-8 + mu)*identity(bl) - Delta[bl][w] - Sigma[bl][w] ) # !!!
  return G

def calc_occupancy(Delta, Sigma, mu):
  Gtrial = calc_G(Delta, Sigma, mu)
  f = interp_A(Gtrial)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    n = integrate.quad(lambda x : 2*f(x)*special.expit(-x/T), -mesh_max, mesh_max)
  return n[0]

def interp_A(G):
  lx = np.array(list(G.mesh.values()))
  ly = sum( sum( -(1.0/math.pi)*np.array(G[bl].data[:,i,i].imag) for i in index_range(G[bl]))
    for bl in G.indices )                                                        
  if normalize_to_one:
    nr = sum( sum( 1 for i in index_range(G[bl]) ) for bl in G.indices )              
    ly = ly/nr                                                                    
  return interpolate.interp1d(lx, ly, kind='cubic', bounds_error=False, fill_value=0)

def save_rho(fn, bgf):
 for bl in bgf.indices:
   with open(fn, "w") as f: 
     for w in bgf[bl].mesh:
       z = bgf[bl][w]
       f.write("%f %f\n" % (w, -1.0/math.pi*z.imag))

def save_gf(fn, bgf):
 for bl in bgf.indices:
   with open(fn, "w") as f: 
     for w in bgf[bl].mesh:
       z = bgf[bl][w]
       f.write("%f %f %f\n" % (w, z.real, z.imag))

def initial_Sigma_mu():
    return new_calculation()

def fmt_str_header(nr_val):
  str = "{:>5}" # itern
  for _ in range(nr_val-1): str += " {:>15}"
  return str + "\n"

def fmt_str(nr_val):
  str = "{:>5}" # itern
  for _ in range(nr_val-1): str += " {:>15.8g}"
  return str + "\n"

def gf_to_nparray(gf):
  return gf.data.flatten()

def bgf_to_nparray(bgf):
  return np.hstack((bgf[bl].data.flatten() for bl in bgf.indices))

def nparray_to_gf(a, gf):
  b = a.reshape(gf.data.shape)
  gf.data[:,:,:] = b[:,:,:]

def nparray_to_bgf(a):
  G = newG()
  split = np.split(a, nr_blocks(G)) 
  for i, bl in enumerate(G.indices):
    nparray_to_gf(split[i], G[bl])
  return G

def gf_diff(a, b):
  f_a = interp_A(a)
  f_b = interp_A(b)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    diff = integrate.quad(lambda x : (f_a(x)-f_b(x))**2, -mesh_max, mesh_max)
  return diff[0]

def compute_G_from_rho(omega, rho_typ):
    Re_G = np.zeros_like(omega)
    for i, w in enumerate(omega):
        integrand = rho_typ / (w - omega + 1j * 1e-4)
        Re_G[i] = simpson(y=np.real(integrand), x=omega)
    Im_G = -np.pi * rho_typ
    return Re_G + 1j * Im_G

#==================================================================================
#This function is only for the initialization
#==================================================================================

def self_consistency0(Sigma):
  Gloc = newG()
  for bl in Gloc.indices:
    for w in Gloc.mesh:
      for i in range(block_size(bl)):
        for j in range(block_size(bl)):
          if i == j:
            new_self = w + U/2 - S.Sigma_w[bl][w][i,i]
            new_v = (V*V)/new_self
            new_vtp = np.sqrt((pow(V,4)/(4 * (new_self**2))) + (tp**2))

            term1 = 0.5 + new_v / (4.0 * new_vtp)
            arg1 = w - 0.5*new_v - new_vtp
            ht_term1 = ht(arg1)

            term2 = 0.5 - new_v / (4.0 * new_vtp)
            arg2 = w - 0.5*new_v + new_vtp
            ht_term2 = ht(arg2)

            Gloc[bl][w][i,i]= (1.0 / new_self) * (1.0 + new_v * (term1 * ht_term1 + term2 * ht_term2))
          else:
            assert abs(Sigma[bl][w][i,j])<1e-10, "This implementation only supports diagonal self-energy"
            Gloc[bl][w][i,j] = 0.0
  Glocinv = Gloc.inverse()
  Delta = newG()
  for bl in Delta.indices:
    for w in Delta.mesh:
      Delta[bl][w] = (w+U/2)*identity(bl) - Sigma[bl][w] - Glocinv[bl][w] 
  return Gloc, Delta

#==================================================================================
def kk_real_from_imag(gf, eps = 1e-10):

  gf_in = gf.copy()
  gf_out = gf.copy()
  for bl in Delta.indices:
    for i in range(block_size(bl)):
      for j in range(block_size(bl)):
        if i == j:
          dosA = Gf(mesh=gf.mesh, target_shape=[])
          for w in gf.mesh:
            dosA[w] = np.array([[ (-1.0/math.pi)*gf_in[w][i,i].imag ]]) # dos = -1/pi Im[gf], ignore real part
          for w in gf.mesh:
            gf_out[w][i,i] = hilbert_transform_refreq(dosA, float(w) + eps*1j)
        else:
          gf_out[w][i,j] = 0
          assert abs(gf_in[w][i,j]) < 1e-16, "This implementation is only valid for diagonal matrix case"
  return gf_out
#==================================================================================
#==================================================================================
#==================================================================================

def fix_hyb_function(Delta, Delta_min):
  Delta_fixed = Delta.copy()
  for bl in Delta.indices:
    for w in Delta.mesh:
      for n in range(block_size(bl)): 
        r = Delta[bl][w][n,n].real
        i = Delta[bl][w][n,n].imag
        #Delta_fixed[bl][w][n,n] = r + 1j*i
        Delta_fixed[bl][w][n,n] = r + 1j*(i if i<-Delta_min else -Delta_min)
    Delta_fixed[bl] << kk_real_from_imag(Delta_fixed[bl])
  return Delta_fixed

#==================================================================================
#Here starts the DMFT self-consistency
#==================================================================================

def dmft_step(Delta_in, disorder_values):
  global itern
  itern += 1
  if verbose:
    print("Iteration %i min_iter=%i max_iter=%i\n" % (itern, min_iter, max_iter))
  Delta_in_fixed = fix_hyb_function(Delta_in, Delta_min)
  S.Delta_w << Delta_in_fixed
 
  global Gself, Gloc, Gloc_prev, rho_arith
  Gself = newG()
  sp = { "T": T, "Lambda": 2.0, "Nz": 4, "Tmin": 1e-5, "keep": 10000, "keepenergy": 10.0 }
  
  mp = { "U1": U, "B1": B, "omega": omega, "g1": g1, "n1": n1 }
  mp = { k: v for k,v in mp.items() if v is not None }
  sp["model_parameters"] = mp
  n_w = len(Gself.mesh)  
  log_rho_sum = np.zeros(n_w) 
  rho_arith_sum = np.zeros(n_w) 

  for i, e_i in enumerate(disorder_values):
    if verbose:
        print("The disorder number and value is: ", i, e_i)
        #print("The disorder value is: ", i)
    e_f = -U/2+e_i
    sp["model_parameters"]["eps1"] = e_f

    S.solve(**sp) 
    for bl in Gself.indices:
        for w in Gself.mesh:
            Gself[bl][w] = np.linalg.inv( (w - e_f)*identity(bl) - Delta_in_fixed[bl][w] - S.Sigma_w[bl][w] ) 
            #Gself[bl][w] = np.linalg.inv( (w + 1j*1e-8 - e_f)*identity(bl) - Delta_in_fixed[bl][w] - S.Sigma_w[bl][w] ) 

    Gself_array = bgf_to_nparray(Gself)
    rho_i = -1.0/math.pi*Gself_array.imag
    rho_i = np.maximum(rho_i, 1e-30)
    log_rho_sum += np.log(rho_i) / len(disorder_values)
    rho_arith_sum += rho_i / len(disorder_values)

  rho_typ = np.exp(log_rho_sum)
  rho_arith = rho_arith_sum
  
  global omega_vals, Glat, Delta_mod, rho_lat, Sigma
  omega_vals = []

  for bl in Gself.indices:
    for w in Gself[bl].mesh:
      omega_vals.append(float(w.real))

  if mpi.is_master_node():
    np.savetxt(f"{itern}_rho_typ.txt", np.column_stack([omega_vals, rho_typ]))
    np.savetxt(f"{itern}_rho_arith.txt", np.column_stack([omega_vals, rho_arith]))

  G_typ = compute_G_from_rho(np.array(omega_vals), rho_typ)
  if mpi.is_master_node():
    np.savetxt(f"{itern}_gtyp.txt", np.column_stack([omega_vals, G_typ.real, G_typ.imag]))

  G_typ_inv = 1.0/G_typ

  Delta_fix = bgf_to_nparray(Delta_in_fixed)

  Sigma = np.array(omega_vals) + U/2.0 - Delta_fix - G_typ_inv
  
  if mpi.is_master_node():
    np.savetxt(f"{itern}_sigma.txt", np.column_stack([omega_vals, Sigma.real, Sigma.imag]))

  new_self = np.array(omega_vals) + U/2 - Sigma
  new_v = (V*V)/new_self
  new_vtp = np.sqrt((pow(V,4)/(4 * (new_self**2))) + (tp**2))

  term1 = 0.5 + new_v / (4.0 * new_vtp)
  arg1 = np.array(omega_vals) - 0.5*new_v - new_vtp
  ht_term1 = ht2(arg1)

  term2 = 0.5 - new_v / (4.0 * new_vtp)
  arg2 = np.array(omega_vals) - 0.5*new_v + new_vtp
  ht_term2 = ht2(arg2)

  Glat = (1.0 / new_self) * (1.0 + new_v * (term1 * ht_term1 + term2 * ht_term2))
  Glat_inv = 1.0/Glat

  if mpi.is_master_node():
    np.savetxt(f"{itern}_glat.txt", np.column_stack([omega_vals, Glat.real, Glat.imag]))

  rho_lat = (-1/math.pi)*Glat.imag
  if mpi.is_master_node():
    np.savetxt(f"{itern}_rho_lat.txt", np.column_stack([omega_vals, rho_lat]))

  Delta_mod = np.array(omega_vals) + U/2 - Sigma - Glat_inv
  if mpi.is_master_node():
    np.savetxt(f"{itern}_delta.txt", np.column_stack([omega_vals, Delta_mod.real, Delta_mod.imag]))

  Delta = nparray_to_bgf(Delta_mod)
  Gloc = nparray_to_bgf(Glat)
  Gtyp = nparray_to_bgf(G_typ)
  Sigma_block = nparray_to_bgf(Sigma)

  #diff_loc_imp = gf_diff(Gself, Gloc)
  diff_loc_imp = gf_diff(Gtyp, Gloc)
  diff_prev = gf_diff(Gloc, Gloc_prev)
  Gloc_prev = Gloc.copy()
  occupancy = calc_occupancy(Delta, Sigma_block, U/2)
  
  stats = OrderedDict([("itern", itern), ("diff_loc_imp", diff_loc_imp), ("diff_prev", diff_prev), ("occupancy", occupancy)])
  #stats = OrderedDict([("itern", itern), ("diff_loc_imp", diff_loc_imp), ("diff_prev", diff_prev), ("occupancy", occupancy)])
  #for i in observables:
  #  stats[i] = S.expv[i]
  header_string = fmt_str_header(len(stats)).format(*[i for i in stats.keys()])
  stats_string  = fmt_str(len(stats)).format(*[i for i in stats.values()])

  if mpi.is_master_node():
    if itern == 1: stats_file.write(header_string)
    stats_file.write(stats_string)
  if verbose: print("stats: %sstats: %s" % (header_string, stats_string)) 

  if (diff_loc_imp   < eps_loc_imp   and
      diff_prev      < eps_prev      and
      itern >= min_iter):
    raise Converged(stats_string)
  if (itern == max_iter):
    raise FailedToConverge(stats_string)
  return Delta

#==================================================================================
#==================================================================================

F = lambda Delta : dmft_step(Delta, disorder_values)-Delta
npF = lambda x : bgf_to_nparray(F(nparray_to_bgf(x)))
def solve_with_broyden_mixing(Delta, alpha):
  xin = bgf_to_nparray(Delta)
  optimize.broyden1(npF, xin, alpha=alpha, reduction_method="svd", max_rank=10, verbose=verbose, f_tol=1e-99) 

Sigma0 = initial_Sigma_mu()
Gloc, Delta = self_consistency0(Sigma0)
Gloc_prev = Gloc.copy()

def solve(Delta):
    solve_with_broyden_mixing(Delta, alpha)

if mpi.is_master_node():
  global stats_file
  stats_file = open("stats.dat", "w", buffering=1) 

if W == 0:
    #N_worker=1
    disorder_values = [0.0]
else:
    #N_worker = N_worker
    disorder_values = generate_disorder(N, W)

try:
  solve(Delta)

except Converged as c:
  if verbose: 
      print("Converged: %s" % c.message)
      np.savetxt("glat_final.txt", np.column_stack([omega_vals, Glat.real, Glat.imag]))
      np.savetxt("rho_lat_final.txt", np.column_stack([omega_vals, rho_lat]))
      np.savetxt("delta_final.txt", np.column_stack([omega_vals, Delta_mod.real, Delta_mod.imag]))
      np.savetxt("rho_arith_final.txt", np.column_stack([omega_vals, rho_arith]))
      np.savetxt("sigma_final.txt", np.column_stack([omega_vals, Sigma.real, Sigma.imag]))

except FailedToConverge as c:
  if verbose: 
      print("Failed to converge: %s" % c.message)
      np.savetxt("glat_final.txt", np.column_stack([omega_vals, Glat.real, Glat.imag]))
      np.savetxt("rho_lat_final.txt", np.column_stack([omega_vals, rho_lat]))
      np.savetxt("delta_final.txt", np.column_stack([omega_vals, Delta_mod.real, Delta_mod.imag]))
      np.savetxt("rho_arith_final.txt", np.column_stack([omega_vals, rho_arith]))
      np.savetxt("sigma_final.txt", np.column_stack([omega_vals, Sigma.real, Sigma.imag]))


mpi.barrier() 
