import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
#Initial code to solve 1D t-indep shcrodinger


def H(Omega, Delta, V, Phi_L):
    H_t_indep = hbar * np.array([
        [2*Delta,                   Omega*np.exp(-1j*Phi_L),    Omega*np.exp(-1j*Phi_L),      0                          ],
        [Omega*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
        [Omega*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
        [0,                         Omega*np.exp(1j*Phi_L),     Omega*np.exp(1j*Phi_L),       -2*Delta + 2*V             ],
        ], dtype=complex)
    return H_t_indep

    
def evolve(H, psi0, t):
    """Returns state vector at time t"""
    eigvals, R_dagger = eigh(H)
    #eigenvecs aka R_dagger is a 4x4 mat whose columns are the eigenvectors

    D = np.diag(np.exp(-1j * eigvals * t / hbar)) # diagonal matrix with
    R = R_dagger.conj().T #rows = conjugated eigenvectors
    
    # See notes for why this convention taken
    return R_dagger @ D @ R @ psi0 #multiplying matrices according to eq 5.6



#placeholder values for now (change to real val)
hbar = 1
Omega = np.pi*10**6
Delta = 0
Phi_L = 1

psi0 = np.array([1, 0, 0, 0], dtype=complex) #initial state in |00>

V_max = 40*Omega
V_values = np.linspace(0, V_max, 500)
#range of V's to show spread


#list of times to look at system (frames of animation)
steps = 200
times = np.linspace(0, 10**-6, steps)


# PLOT
fig, ax = plt.subplots(figsize=(10, 3))

#indexing for 1 till end of array bc first one should be blue (Fig 5.1)
for V in V_values[1:]:
    psi_t = np.array([evolve(H(Omega, Delta, V, Phi_L), psi0, t) for t in times])
    #Scale time axis *1e6
    #abs**2 is probability
    alpha = 0.05 + 0.95 * (V / V_max) #colour gets darker for larger V
    ax.plot(times*1e6, np.abs(psi_t[:, 0])**2, color="mediumvioletred", alpha = alpha, lw=1)


psi_t = np.array([evolve(H(Omega, Delta, 0, Phi_L), psi0, t) for t in times])
ax.plot(times*1e6, np.abs(psi_t[:, 0])**2, color="c", lw = 2)


ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Time (µs)")
ax.set_ylabel(r"$P_{|00\rangle}$")
plt.show()