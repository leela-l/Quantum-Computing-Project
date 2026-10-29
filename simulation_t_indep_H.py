import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
#Initial code to solve 1D t-indep shcrodinger


def H(V):
    H_matrix= hbar * np.array([
        [2*Delta,                   Omega*np.exp(-1j*Phi_L),    Omega*np.exp(-1j*Phi_L),      0                          ],
        [Omega*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
        [Omega*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
        [0,                         Omega*np.exp(1j*Phi_L),     Omega*np.exp(1j*Phi_L),       -2*Delta + 2*V             ],
        ], dtype=complex)
    return H_matrix

    
def evolve(H, psi0, t):
    """Returns state vector at time t"""
    eigvals, R_dagger = eigh(H)
    #eigenvecs aka R_dagger is a 4x4 mat whose columns are the eigenvectors

    D = np.diag(np.exp(-1j * eigvals * t / hbar)) # diagonal matrix with
    R = R_dagger.conj().T #rows = conjugated eigenvectors
    
    # See notes for why this convention taken
    return R_dagger @ D @ R @ psi0 #multiplying matrices according to eq 5.6


def bell_fidelity(psi):
    return np.abs(psi_t @ bell_state.conj())**2


#placeholder values for now (change to real val)
hbar = 1
Omega = np.pi*10**6
Delta = 0
Phi_L = 1

psi0 = np.array([1, 0, 0, 0], dtype=complex) #initial state in |00>
bell_state = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)   # (|01> + |10>)/sqrt2

V_max = 40*Omega
V_values = np.linspace(0, V_max, 500)
#range of V's to show spread


#list of times to look at system (frames of animation)
steps = 200
times = np.linspace(0, 10**-6, steps)


# PLOT
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,5), sharex=True)

psi_t = np.array([evolve(H(0), psi0, t) for t in times])
ax1.plot(times*1e6, np.abs(psi_t[:, 0])**2, color="c", lw = 2)
ax2.plot(times*1e6, bell_fidelity(psi_t), color="c", lw = 2)



#indexing for 1 till end of array bc first one should be blue (Fig 5.1)
for V in V_values[1:]:
    psi_t = np.array([evolve(H(V), psi0, t) for t in times])
    #Scale time axis *1e6
    #abs**2 is probability
    alpha = 0.05 + 0.95 * (V / V_max) #colour gets darker for larger V
    ax1.plot(times*1e6, np.abs(psi_t[:, 0])**2, color="mediumvioletred", alpha = alpha, lw=1)
    ax2.plot(times*1e6, bell_fidelity(psi_t), color="mediumvioletred", alpha = alpha, lw=1)



ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax2.set_xlabel("Time (µs)")
ax1.set_ylabel(r"$P_{|00\rangle}$")
ax2.set_ylabel(r"$/psi^+$")
plt.show()