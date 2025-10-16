import numpy as np
import matplotlib.pyplot as plt
from boundary import ForwardProblem, AdjointProblem
from PGS import ProjectionGradientSolver

l, T = 2.0, 0.5
a2, nu = 0.1, 1.0
N, M = 200, 200
R = 10.0

max_iters = 1000

# func = lambda x: np.power(x, 1)

h, tau = l / N, T / M
s = np.linspace(0, l, N + 1)
phi = np.zeros_like(s)
# y_target = np.cos(np.pi * s / l)
# y_target = func(s)

# y_target = np.cos(np.pi * s / l)
# y_target = np.sin(np.pi * s / l) ** 2
y_target = np.exp(-0.5 * s) * np.cos(np.pi * s / l)
# y_target = np.cos(np.pi * s / l)**3
# y_target = 1 - 2 / (1 + np.exp(-10 * (0.5 - s / l)))
# y_target = np.tanh(5 * (0.5 - s / l))
# y_target = 1 - 4 * (s / l - 0.5)**3

forward_problem = ForwardProblem(a2, nu, l, T, N, M, phi)
adjoint_problem = AdjointProblem(a2, nu, l, T, N, M, y_target)
solver = ProjectionGradientSolver(forward_problem, adjoint_problem, R, phi)

f0 = np.zeros((N + 1, M + 1))
print("=== Backtracking ===")
f_back, hist_back = solver.solve(
    f0, max_iters=max_iters, alpha_strategy="backtracking", alpha0=1e-1
)
print("\n=== Lipschitz estimate ===")
f_lip, hist_lip = solver.solve(
    f0, max_iters=max_iters, alpha_strategy="lipschitz_est", alpha0=1e-2
)

x_back = forward_problem.solve(f_back)
x_lip = forward_problem.solve(f_lip)

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(s, x_back[:, -1], label="$x$ (backtracking)", color="fuchsia")
plt.plot(s, x_lip[:, -1], label="$x$ (lipschitz)", color="cyan")
plt.plot(s, y_target, "--", label="$y$", color="purple")
plt.xlabel("$s$")
plt.title("Temperature at $T$")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.semilogy(hist_back["J"], label="$J$ (backtracking)", color="royalblue")
plt.semilogy(hist_lip["J"], label="$J$ (lipschitz)", color="violet")
plt.xlabel("Iteration")
plt.title("Convergence of $J$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
