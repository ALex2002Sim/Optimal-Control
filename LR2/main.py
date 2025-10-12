# main_wave.py
import numpy as np
import matplotlib.pyplot as plt
from boundary import WaveSolver
from CGS import ConditionalGradientSolver

# ПАРАМЕТРЫ (можешь менять для экспериментов)
l, T = 2.0, 1.0
a2 = 1.0
N, M = 200, 400   # дискретизация: уменьши M если очень медленно
R0 = 5.0

# сетка и начальные условия
wave = WaveSolver(a2=a2, l=l, T=T, N=N, M=M)
s = wave.s
t = wave.t

phi0 = np.zeros_like(s)               # x(s,0)
phi1 = np.zeros_like(s)               # x_t(s,0)
# пример целевой формы y(s)
y_target = 0.5 * np.sin(np.pi * s / l)

# внешний источник ноль
f_mat = np.zeros((N + 1, M + 1))

# начальный  p0 (нулевой)
p0 = np.zeros(M + 1)

solver = ConditionalGradientSolver(wave, R0, phi0, phi1, y_target, f_mat)

# Запуск: способ 1 — line search (номер 1 в методичке)
p_line, hist_line = solver.solve(p0, max_iters=1000, tol=1e-4, alpha_strategy="line_search")

# Запуск: способ 5 — apriori sequence
p0 = np.zeros(M + 1)
p_apr, hist_apr = solver.solve(p0, max_iters=1000, tol=1e-4, alpha_strategy="apriori")

# Построим итоговые распределения в конце
x_line = wave.forward_solve(p_line, f_mat, phi0, phi1)
x_apr = wave.forward_solve(p_apr, f_mat, phi0, phi1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(s, x_line[:, -1], label='x (line-search)')
plt.plot(s, x_apr[:, -1], label='x (apriori)')
plt.plot(s, y_target, '--', label='y target')
plt.xlabel('s'); plt.title('x(s,T)')
plt.legend(); plt.grid()

plt.subplot(1, 2, 2)
plt.plot(hist_line['J'], label='J (line-search)')
plt.plot(hist_apr['J'], label='J (apriori)')
plt.yscale('log')
plt.xlabel('iter'); plt.title('J convergence')
plt.legend(); plt.grid()
plt.tight_layout()
plt.show()
