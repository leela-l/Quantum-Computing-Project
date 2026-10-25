import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
#Initial code to solve 1D t-indep shcrodinger

def evolve_Delta(Delta0, t):
    """Linear sweep through resonance"""
    return Delta0


def H(Omega, Delta, V, Phi_L):
    H_matrix= hbar * np.array([
        [2*Delta,                   Omega*np.exp(-1j*Phi_L),    Omega*np.exp(-1j*Phi_L),      0                          ],
        [Omega*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
        [Omega*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
        [0,                         Omega*np.exp(1j*Phi_L),     Omega*np.exp(1j*Phi_L),       -2*Delta + 2*V             ],
        ], dtype=complex)
    return H_matrix


def step(Omega, Delta0, Phi_L, psi, t, dt):
    """Evolve psi from t to t+dt, assuming H as const over the step."""
    D_intermediate = evolve_Delta(Delta0, t+dt/2)
    # Using dt/2 rather than dt makes error per step smaller
    eigvals, R_dagger = eigh(H(Omega, D_intermediate, V, Phi_L)) # decided function is not needed
    D = np.diag(np.exp(-1j * eigvals * dt / hbar))
    return R_dagger @ D @ R_dagger.conj().T @ psi


#placeholder values for now (change to real val)
hbar = 1
Omega = np.pi*10**6
Delta0 = 5*Omega
V = 2*np.pi*10*1e6 
Phi_L = 1

psi0 = np.array([1, 0, 0, 0], dtype=complex) #initial state in |00>




steps = 2000
t_max = 1e-6
times = np.linspace(0, t_max, steps + 1)
dt = times[1] - times[0] #time between steps

psi = psi0.copy() #Add .copy() just for saftey in future
probs = np.zeros((steps + 1, 4)) #defines empty array to fill during iterations
probs[0] = np.abs(psi)**2
bell_probs = np.zeros((steps + 1, 4)) #defines empty array to fill during iterations
bell_probs[0] = 0

for i in range(steps):
    psi = step(Omega, V, Phi_L, psi, times[i], dt)
    probs[i + 1] = np.abs(psi)**2 #adds new prob to array
    probs[i+1] = np.abs(psi)**2


#[: (all rows), x (in this column)]
fig, ax = plt.subplots()
ax.plot(times*1e6, probs[:, 0], label=r"$P_{|00\rangle}$")
ax.set_xlabel("Time (µs)"); ax.set_ylabel("Probability"); ax.legend()
plt.show()