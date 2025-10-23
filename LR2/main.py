import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

plt.rcParams["figure.figsize"] = (8, 5)

from scipy.optimize import minimize_scalar


def line_search_brent(p_k, p_bar, expand_factor=3.0):

    def J_of_alpha(alpha):
        p_try = p_k + alpha * (p_bar - p_k)
        return compute_J_from_p(p_try)

    alphas_test = np.linspace(0, 1.5, 6)
    J_vals = [J_of_alpha(a) for a in alphas_test]
    i_min = int(np.argmin(J_vals))

    a_left = 0.0
    a_right = 1.0
    if i_min == len(alphas_test) - 1:
        a_right *= expand_factor

    res = minimize_scalar(
        J_of_alpha, bounds=(a_left, a_right), method="bounded", options={"xatol": 1e-3}
    )

    if not res.success or abs(res.x) < 1e-7:
        return 0.5
    return float(res.x)


a = 1.0
l = 1.0
T = 1.0
N = 100
M = 200
h = l / N
tau = T / M

mu = a**2 * tau**2 / (2 * h**2)
if mu <= 0:
    raise RuntimeError("mu nonpositive")
print(f"Используем параметры: N={N}, M={M}, h={h:.4e}, tau={tau:.4e}, mu={mu:.4e}")

s_grid = np.linspace(0, l, N + 1)
t_grid = np.linspace(0, T, M + 1)


def phi0(s):
    return np.zeros_like(s)


def phi1(s):
    return np.zeros_like(s)


def y_target(s):
    return np.sin(np.pi * s / l) * np.exp(np.power(s, 2))
    # return np.power((s - l), 2)


R0 = 1000.0


def solve_forward(p_time):
    X = np.zeros((N + 1, M + 1))
    X[:, 0] = phi0(s_grid)
    D2_0 = X[0:-2, 0] - 2 * X[1:-1, 0] + X[2:, 0]
    X[1:-1, 1] = X[1:-1, 0] + tau * phi1(s_grid[1:-1]) + mu * D2_0
    X[0, 0] = X[1, 0] - h * p_time[0]
    X[0, 1] = X[1, 1] - h * p_time[1]
    X[-1, 0] = X[-2, 0]
    X[-1, 1] = X[-2, 1]
    for j in range(1, M):
        rhs = (
            2 * X[1:-1, j]
            - X[1:-1, j - 1]
            + mu * (X[0:-2, j - 1] - 2 * X[1:-1, j - 1] + X[2:, j - 1])
        )
        A_diag = np.ones(N - 1) * (1 + 2 * mu)
        A_low = np.ones(N - 2) * (-mu)
        A_up = np.ones(N - 2) * (-mu)
        A_diag[0] = 1 + mu
        rhs[0] = rhs[0] + mu * h * p_time[j + 1]
        A_diag[-1] = 1 + mu
        ab = np.zeros((3, N - 1))
        ab[0, 1:] = A_up
        ab[1, :] = A_diag
        ab[2, :-1] = A_low
        x_inner = solve_banded((1, 1), ab, rhs)
        X[1:-1, j + 1] = x_inner
        X[0, j + 1] = X[1, j + 1] - h * p_time[j + 1]
        X[-1, j + 1] = X[-2, j + 1]
    return X


def solve_adjoint(x_forward):
    PSI = np.zeros((N + 1, M + 1))
    PSI[:, M] = 0.0
    PSI_t_at_T = 2.0 * (x_forward[:, M] - y_target(s_grid))
    D2_M = PSI[0:-2, M] - 2 * PSI[1:-1, M] + PSI[2:, M]
    PSI[1:-1, M - 1] = PSI[1:-1, M] - tau * PSI_t_at_T[1:-1] + mu * D2_M
    PSI[0, M] = PSI[1, M]
    PSI[-1, M] = PSI[-2, M]
    PSI[0, M - 1] = PSI[1, M - 1]
    PSI[-1, M - 1] = PSI[-2, M - 1]
    for j in range(M - 1, 0, -1):
        rhs = (
            2 * PSI[1:-1, j]
            - PSI[1:-1, j + 1]
            + mu * (PSI[0:-2, j + 1] - 2 * PSI[1:-1, j + 1] + PSI[2:, j + 1])
        )
        A_diag = np.ones(N - 1) * (1 + 2 * mu)
        A_low = np.ones(N - 2) * (-mu)
        A_up = np.ones(N - 2) * (-mu)
        A_diag[0] = 1 + mu
        A_diag[-1] = 1 + mu
        ab = np.zeros((3, N - 1))
        ab[0, 1:] = A_up
        ab[1, :] = A_diag
        ab[2, :-1] = A_low
        psi_inner = solve_banded((1, 1), ab, rhs)
        PSI[1:-1, j - 1] = psi_inner
        PSI[0, j - 1] = PSI[1, j - 1]
        PSI[-1, j - 1] = PSI[-2, j - 1]
    return PSI


def compute_J_from_p(p_time):
    X = solve_forward(p_time)
    diff = X[:, -1] - y_target(s_grid)
    return np.trapezoid(diff**2, s_grid)


p0 = np.zeros(M + 1)

max_iter = 500
tol = 1e-8


def run_conditional_gradient(alpha_strategy):
    p_k = p0.copy()
    errors = []
    p_history = []
    for k in range(max_iter):
        Xk = solve_forward(p_k)
        PSI = solve_adjoint(Xk)
        psi0_time = PSI[0, :]
        norm_psi0 = np.sqrt(np.trapezoid(psi0_time**2, t_grid))
        if norm_psi0 < 1e-14:
            p_bar = np.zeros_like(p_k)
        else:
            p_bar = -R0 * psi0_time / norm_psi0
        if alpha_strategy == "line_search":
            alpha_k = line_search_brent(p_k, p_bar)
        elif alpha_strategy == "preset_1_over_k":
            alpha_k = 1.0 / (k + 1)
        else:
            raise ValueError("unknown strategy")
        p_k1 = p_k + alpha_k * (p_bar - p_k)
        Xk1 = solve_forward(p_k1)
        err = np.sqrt(np.trapezoid((Xk1[:, -1] - y_target(s_grid)) ** 2, s_grid))
        errors.append(err)
        p_history.append(p_k1.copy())
        if k > 0 and abs(errors[-1] - errors[-2]) < tol:
            break
        p_k = p_k1
        if k % 5 == 0:
            print(f"[{alpha_strategy}] iter {k}, alpha={alpha_k:.4e}, err={err:.6e}")
    return errors, p_history


errors_line, p_hist_line = run_conditional_gradient("line_search")
errors_preset, p_hist_preset = run_conditional_gradient("preset_1_over_k")

plt.figure()
plt.plot(
    np.arange(1, len(errors_line) + 1), errors_line, label="line search (method 1)"
)
plt.plot(
    np.arange(1, len(errors_preset) + 1),
    errors_preset,
    label="alpha = 1/(k+1) (method 5)",
)
plt.xlabel("итерация k")
plt.ylabel("||x(.,T)-y||_2")
plt.title("Сходимость по итерациям (демонстрация)")
plt.grid(True)
plt.legend()
plt.show()

X_line_final = solve_forward(p_hist_line[-1] if p_hist_line else p0)
X_preset_final = solve_forward(p_hist_preset[-1] if p_hist_preset else p0)

plt.figure()
plt.plot(s_grid, X_line_final[:, -1], label="x_T (line search)")
plt.plot(s_grid, X_preset_final[:, -1], label="x_T (preset 1/(k+1))")
plt.plot(s_grid, y_target(s_grid), "--", label="y(s) target")
plt.xlabel("s")
plt.ylabel("x(s,T)")
plt.title("Конечные профили струны в момент T (демонстрация)")
plt.legend()
plt.grid(True)
plt.show()

J_line = compute_J_from_p(p_hist_line[-1]) if p_hist_line else compute_J_from_p(p0)
J_preset = (
    compute_J_from_p(p_hist_preset[-1]) if p_hist_preset else compute_J_from_p(p0)
)
print(f"Итоги: J(line_search)={J_line:.6e}, J(preset)= {J_preset:.6e}")
