# boundary_wave.py
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class TripleDiagMatrix:
    """Классический метод прогонки для трёхдиагональной СЛАУ"""

    def _thomas_solve(
        self,
        lower: NDArray[np.float64],
        main: NDArray[np.float64],
        upper: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        N = len(main)
        cp = np.zeros(N, dtype=np.float64)
        dp = np.zeros(N, dtype=np.float64)
        x = np.zeros(N, dtype=np.float64)

        cp[0] = upper[0] / main[0]
        dp[0] = b[0] / main[0]

        for i in range(1, N):
            denom = main[i] - lower[i - 1] * cp[i - 1]
            cp[i] = upper[i] / denom if i < N - 1 else 0.0
            dp[i] = (b[i] - lower[i - 1] * dp[i - 1]) / denom

        x[-1] = dp[-1]
        for i in range(N - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x


class WaveSolver(TripleDiagMatrix):
    """
    Неявная схема для уравнения колебаний:
      x_t = a^2 * x_ss + f(s,t)
    (теплопроводность/волновое уравнение первого порядка по времени)
    Управление p(t): левое условие x_s(0)=p(t), правое Neumann: x_s(l)=0.
    """

    def __init__(self, a2: float, l: float, T: float, N: int, M: int):
        self.a2 = a2
        self.l = l
        self.T = T
        self.N = N
        self.M = M
        self.h = l / N
        self.tau = T / M
        self.mu = self.a2 * self.tau / (self.h ** 2)

        self.s = np.linspace(0, l, N + 1)
        self.t = np.linspace(0, T, M + 1)

        # строим трёхдиагональную матрицу (с учётом граничных условий)
        self.lower_diag, self.main_diag, self.upper_diag = self._build_diagonals()

    def _build_diagonals(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        N = self.N
        mu = self.mu

        main_diag = np.ones(N + 1) * (1.0 + 2.0 * mu)
        lower_diag = np.full(N, -mu)
        upper_diag = np.full(N, -mu)

        # Левая граница (x_0 = x_1 - h*p): скорректируем первую строку
        main_diag[0] = 1.0
        upper_diag[0] = -1.0

        # Правая граница (x_N = x_{N-1})
        main_diag[-1] = 1.0 + mu
        lower_diag[-1] = -mu

        return lower_diag, main_diag, upper_diag

    def forward_solve(
        self,
        p: NDArray[np.float64],
        f_mat: NDArray[np.float64],
        phi0: NDArray[np.float64],
        phi1: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Решение прямой задачи методом неявной схемы.
        p(t): управление, задаёт левое условие x_s(0)=p(t)
        phi0: начальное распределение x(s,0)
        phi1: x_t(s,0) — не используется (можно оставить ноль)
        """
        N, M = self.N, self.M
        h, tau, mu = self.h, self.tau, self.mu

        x = np.zeros((N + 1, M + 1))
        x[:, 0] = phi0.copy()

        for j in range(M):
            b = x[:, j].copy() + tau * f_mat[:, j + 1]

            # Граничные условия:
            b[0] = 0.0                      # слева x0 = x1 - h*p -> реализовано в матрице
            b[-1] += mu * h * p[j + 1]      # добавка от xN = x_{N-1} учтена неявно

            # учёт лев. управления (x0=x1-hp): правая часть корректируется:
            b[1] += mu * h * p[j + 1]

            # решение через прогонку
            x[:, j + 1] = self._thomas_solve(self.lower_diag, self.main_diag, self.upper_diag, b)

        return x

    def adjoint_solve(self, x_all: NDArray[np.float64], y_target: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Сопряжённая задача для неявной схемы (обратный проход).
        Схема: (I - μ A^T) ψ^{j} = ψ^{j+1} + τ * rhs,
        ψ^M = 2(x^M - y_target)
        """
        N, M = self.N, self.M
        psi = np.zeros_like(x_all)
        h, tau = self.h, self.tau

        # начальные (финальные) условия
        psi[:, M] = 2.0 * (x_all[:, M] - y_target)

        for j in range(M - 1, -1, -1):
            b = psi[:, j + 1].copy()
            b[0] = 0.0
            psi[:, j] = self._thomas_solve(self.lower_diag, self.main_diag, self.upper_diag, b)

        return psi
