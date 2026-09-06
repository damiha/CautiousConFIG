import numpy as np
import matplotlib.pyplot as plt

N_x = 200
N_y = 200

x = np.linspace(-0.5, 1, N_x)
y = np.linspace(-0.5, 1.5, N_y)

Y, X = np.meshgrid(y, x, indexing="ij")

nu = 1 / 40
lambda_val = 1 / (2 * nu) - np.sqrt(1 / (4 * nu**2) + 4 * np.pi**2)

u_x = 1 - np.exp(lambda_val * X) * np.cos(2 * np.pi * Y)

u_y = lambda_val / (2 * np.pi) * np.exp(lambda_val * X) * np.sin(2 * np.pi * Y)

p = 0.5 * (1 - np.exp(2 * lambda_val * X))


plt.title("u_x")
plt.imshow(u_x, origin="lower", extent=[-0.5, 1, -0.5, 1.5], cmap="coolwarm")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("u_x.png", dpi=300)
plt.close()


plt.title("u_y")
plt.imshow(u_y, origin="lower", extent=[-0.5, 1, -0.5, 1.5], cmap="coolwarm")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("u_y.png", dpi=300)
plt.close()


plt.title("p")
plt.imshow(p, origin="lower", extent=[-0.5, 1, -0.5, 1.5], cmap="coolwarm")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("p.png", dpi=300)
plt.close()

#np.savez_compressed("kovasznay_analytical_solution.npz", u_x=u_x, u_y = u_y, p=p)