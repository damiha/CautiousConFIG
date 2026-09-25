import numpy as np
import matplotlib.pyplot as plt

L = 10.0

# good for fft
N = 256
x = np.linspace(-5.0, 5.0, N, endpoint=False)
dx = x[1] - x[0]

dt = 0.001
t_max = np.pi / 2
steps = int(t_max / dt)

# how?
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

h = 2.0 / np.cosh(x) + 0j

lin_operator = np.exp(-1j * k**2 * dt / 4)

# Zeitschleife (Strang-Splitting)
hs_total = [np.copy(h)]
t_curr = 0

for _ in range(steps):
    
    # half-linear
    h = np.fft.ifft(lin_operator * np.fft.fft(h))

    # full-nonlinear
    nonlin_operator = np.exp(1j * np.abs(h)**2 * dt)

    h = nonlin_operator * h

    # half-linear
    h = np.fft.ifft(lin_operator * np.fft.fft(h))

    hs_total.append(np.copy(h))
    t_curr += dt

#print("final time")
#print(t_curr)
hs = np.array(hs_total)

np.savez_compressed("schroedinger_numerical_solution.npz", h=hs)

'''
t_selected = 0.79
t_step_selected = int(t_selected / dt)

hs = np.array(hs_total)

print(hs.dtype)

hs_selected = hs[t_step_selected, :]

plt.plot(x, np.absolute(hs_selected))
plt.xlim(-5, 5)
plt.ylim(0, 6)
plt.title(f"t = {t_selected}")
plt.xlabel("x")
plt.ylabel("|h|")
plt.savefig("schrödinger_mag.png")
'''

u = hs.real
v = hs.imag
mag_h = np.absolute(hs)


plt.figure(figsize=(5, 5))
plt.imshow(u, extent=[0, np.pi / 2, -5, 5], origin="lower", aspect="auto", cmap="coolwarm")
plt.xlabel("t")
plt.ylabel("x")
plt.colorbar(location="top", pad=0.08)
plt.savefig("schroedinger_u_vals.png", bbox_inches='tight', dpi=300)



plt.figure(figsize=(5, 5))
plt.imshow(v, extent=[0, np.pi / 2, -5, 5], origin="lower", aspect="auto", cmap="coolwarm")
plt.xlabel("t")
plt.ylabel("x")
plt.colorbar(location="top", pad=0.08)
plt.savefig("schroedinger_v_vals.png", bbox_inches='tight', dpi=300)


plt.figure(figsize=(5, 5))
plt.imshow(mag_h, extent=[0, np.pi / 2, -5, 5], origin="lower", aspect="auto", cmap="Reds")
plt.xlabel("t")
plt.ylabel("x")
plt.colorbar(location="top", pad=0.08)
plt.savefig("schroedinger_h_mag_vals.png", bbox_inches='tight', dpi=300)
