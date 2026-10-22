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


def H(t):
    Omega_t = Omega * np.cos(2*np.pi*1e5*t)
    H_t_dep = hbar * np.array([
    [2*Delta,                   Omega_t*np.exp(-1j*Phi_L),    Omega_t*np.exp(-1j*Phi_L),      0                          ],
    [Omega_t*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
    [Omega_t*np.exp(1j*Phi_L),    0,                          0,                            Omega*np.exp(-1j*Phi_L)    ],
    [0,                         Omega_t*np.exp(1j*Phi_L),     Omega_t*np.exp(1j*Phi_L),       -2*Delta + 2*V             ],
    ], dtype=complex)
    return H_t_dep


def step(psi, t, dt):
    """Evolve psi from t to t+dt, assuming H as const over the step."""
    H_intermediate = H(t+dt/2)
    # Using dt/2 rather than dt makes error per step smaller
    eigvals, R_dagger = eigh(H_intermediate) # decided function is not needed
    D = np.diag(np.exp(-1j * eigvals * dt / hbar))
    return R_dagger @ D @ R_dagger.conj().T @ psi


steps = 2000
t_max = 1e-6
times = np.linspace(0, t_max, steps + 1)
dt = times[1] - times[0] #time between steps

psi = psi0.copy() #Add .copy() just for saftey in future
probs = np.zeros((steps + 1, 4)) #defines empty array to fill during iterations
probs[0] = np.abs(psi)**2

for i in range(steps):
    psi = step(psi, times[i], dt)
    probs[i + 1] = np.abs(psi)**2 #adds new prob to array


print(probs)


#[: (all rows), x (in this column)]
plt.plot(times, probs[:, 0], label="|00>")


plt.xlabel("time")
plt.ylabel("probability")
plt.show()