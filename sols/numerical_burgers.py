import numpy as np
import matplotlib.pyplot as plt

N_x = 200
N_t = 200

# 1 - (-1) = 2
l_x = 2

# t in [0, 1]
l_t = 1

delta_t = l_t / (N_t - 1)
delta_x = l_x / (N_x - 1)

assert delta_t <= delta_x, f"CFL condition violated: {delta_t=}, {delta_x=}"

u = np.zeros(shape=(N_x, N_t))

x_i = delta_x * np.arange(N_x) - 1

u[:, 0] = -np.sin(np.pi * x_i)

visc = 0.01 / np.pi

for j in range(0, N_t - 1):

    # in the interior
    for i in range(1, N_x - 1):

        # for u_i, (j + 1)
        ddu_dxx = (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / (delta_x**2)

        if u[i, j] > 0:
            
            V = (u[i, j] - u[i - 1, j]) / delta_x
        else:
            V = (u[i + 1, j] - u[i, j]) / delta_x

        u[i, j + 1] = u[i, j] + delta_t * (visc * ddu_dxx - V * u[i, j])

plt.figure(figsize=(5, 5))
plt.imshow(u, extent=[0, 1, -1,  1], origin="lower", aspect="auto", cmap="coolwarm")
plt.colorbar()
plt.xlabel("t")
plt.ylabel("x")
plt.savefig("burgers_solution.png", dpi=300, bbox_inches='tight')

filename = "burgers_numerical_solution.npz"
np.savez_compressed(filename, u=u)
print(f"Numerical solution saved under {filename}")
