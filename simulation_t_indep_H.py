import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
#Initial code to solve 1D t-indep shcrodinger


#placeholder values for now (change to real val)
hbar = 1
Omega = np.pi*10**6
Delta = 0
V = 2*np.pi*10*1e6 
Phi_L = 1

psi0 = np.array([1, 0, 0, 0], dtype=complex) #initial state in |00>


H_t_indep = hbar * np.array([
    [2*Delta,                   Omega*np.exp(-1j*Phi_L),    Omega*np.exp(-1j*Phi_L),      0                          ],
    [Omega*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
    [Omega*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
    [0,                         Omega*np.exp(1j*Phi_L),     Omega*np.exp(1j*Phi_L),       -2*Delta + 2*V             ],
], dtype=complex)

def find_eigenvec_vals(mat): #input must be np array
    eigvals, eigvecs = eigh(mat)
    #eigenvec is a 4x4 mat whose columns are the eigenvectors
    return eigvals, eigvecs
    
def evolve(R_dagger, eigvals, t):
    "Returns state vector at time t"
    D = np.diag(np.exp(-1j * eigvals * t / hbar)) # diagonal matrix with
    R = R_dagger.conj().T #rows = conjugated eigenvectors
    # See notes for why this convention taken
    return R_dagger @ D @ R @ psi0 #multiplying matrices according to eq 5.6

eigvals, eigvecs = find_eigenvec_vals(H_t_indep)


#list of times to look at system (frames of animation)
#np.linspace(start, stop, count)
times = np.linspace(0, 10**-6, 200)
#abs is magnitude of cmplx number
#probability is amplitude^2
probs= np.array([np.abs(evolve(eigvecs, eigvals, t))**2 for t in times])
print(probs)


#[: (all rows), x (in this column)]
plt.plot(times, probs[:, 0], label="|00>")


plt.xlabel("time")
plt.ylabel("probability")
plt.show()