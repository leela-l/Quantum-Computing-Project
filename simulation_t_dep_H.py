import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
#Initial code to solve 1D t-indep shcrodinger


def Delta(t):
    return 0.0*t


def Omega(delta):
    return np.pi * 1e6 + 0 * delta


def H(V, t):
    delta = Delta(t)
    omega = Omega(delta)
    H_matrix= hbar * np.array([
        [2*delta,                   omega*np.exp(-1j*Phi_L),    omega*np.exp(-1j*Phi_L),      0                          ],
        [omega*np.exp(1j*Phi_L),    0,                          0,                            omega*np.exp(-1j*Phi_L)    ],
        [omega*np.exp(1j*Phi_L),    0,                          0,                            omega*np.exp(-1j*Phi_L)    ],
        [0,                         omega*np.exp(1j*Phi_L),     omega*np.exp(1j*Phi_L),       -2*delta + 2*V             ],
        ], dtype=complex)
    return H_matrix


def step(H_matrix, psi, dt):
    eigvals, R_dagger = eigh(H_matrix)
    D = np.diag(np.exp(-1j * eigvals * dt / hbar)) # diagonal matrix with
    R = R_dagger.conj().T #rows = conjugated eigenvectors
    return R_dagger @ D @ R @ psi


def evolve(times, V):
    psi = psi0.copy()
    states = [psi0]

    for i in range(len(times)-1):
        t_mid = times[i] + dt/2
        psi = step(H(V, t_mid), psi, dt)
        states.append(psi)

    return np.array(states)




def bell_fidelity(psi_t):
    return np.abs(psi_t @ bell_state.conj())**2



Omega0= np.pi*10**6
Delta0 = 0
Phi_L = 1
hbar = 1

psi0 = np.array([1, 0, 0, 0], dtype=complex) #initial state in |00>
bell_state = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)   # (|01> + |10>)/sqrt2


V_max = 40*Omega0
V_values = np.linspace(0, V_max, 500)
#range of V's to show spread


#list of times to look at system (frames of animation)
steps = 200
times = np.linspace(0, 10**-6, steps)

dt = times[1]-times[0]

# PLOT
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,5), sharex=True)

psi_t = evolve(times, 0)
ax1.plot(times*1e6, np.abs(psi_t[:, 0])**2, color="c", lw = 2)
ax2.plot(times*1e6, bell_fidelity(psi_t), color="c", lw = 2)


for V in V_values[1:]:
    psi_t = evolve(times, V)
    alpha = 0.05 + 0.95 * (V / V_max) #colour gets darker for larger V
    ax1.plot(times*1e6, np.abs(psi_t[:, 0])**2, color="mediumvioletred", alpha = alpha, lw=1)
    ax2.plot(times*1e6, bell_fidelity(psi_t), color="mediumvioletred", alpha = alpha, lw=1)



ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

ax2.set_xlabel("Time (µs)")
ax1.set_ylabel(r"$P_{|00\rangle}$")
ax2.set_ylabel(r"$\psi^+$")
plt.show()