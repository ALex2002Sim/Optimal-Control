import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class TripleDiagMatrix:
    def _thomas_solve(
        self,
        lower: NDArray[np.float64],
        main: NDArray[np.float64],
        upper: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        N = len(main) - 1
        cp = np.zeros(N + 1, dtype=np.float64)
        dp = np.zeros(N + 1, dtype=np.float64)
        x = np.zeros(N + 1, dtype=np.float64)

        cp[0] = upper[0] / main[0]
        dp[0] = b[0] / main[0]
        for i in range(1, N + 1):
            denom = main[i] - lower[i - 1] * cp[i - 1]
            cp[i] = upper[i] / denom if i < N else 0.0
            dp[i] = (b[i] - lower[i - 1] * dp[i - 1]) / denom

        x[N] = dp[N]
        for i in range(N - 1, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]

        return x

    def _build_diagonals(
        self, N: int, h: float, tau: float, a2: float, nu: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        mu = a2 * tau / (h * h)

        main_diag = np.ones(N + 1) * (1.0 + 2.0 * mu)
        lower_diag = np.full(N, -mu)
        upper_diag = np.full(N, -mu)

        main_diag[0], upper_diag[0] = 1.0, -1.0
        main_diag[-1] = 1.0 + mu * (1.0 + nu * h)
        lower_diag[-1] = -mu

        return lower_diag, main_diag, upper_diag


class PDEProblem(TripleDiagMatrix):
    def __init__(
        self, a2: float, nu: float, l: float, T: float, N: int, M: int
    ) -> None:
        self.a2 = a2
        self.nu = nu
        self.l = l
        self.T = T
        self.N = N
        self.M = M
        self.h = l / N
        self.tau = T / M
        self.s = np.linspace(0, l, N + 1)
        self.t = np.linspace(0, T, M + 1)
        self.lower_diag, self.main_diag, self.upper_diag = self._build_diagonals(
            self.N, self.h, self.tau, self.a2, self.nu
        )

    def _solve(self, b: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._thomas_solve(self.lower_diag, self.main_diag, self.upper_diag, b)


class ForwardProblem(PDEProblem):
    def __init__(
        self,
        a2: float,
        nu: float,
        l: float,
        T: float,
        N: int,
        M: int,
        phi: NDArray[np.float64],
        p_fixed: float = 0.0,
    ) -> None:
        super().__init__(a2, nu, l, T, N, M)
        self.phi = phi
        self.p_fixed = p_fixed

    def solve(self, f_mat: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.zeros((self.N + 1, self.M + 1))
        x[:, 0] = self.phi.copy()
        mu = self.a2 * self.tau / (self.h * self.h)

        for j in range(self.M):
            b = x[:, j].copy()
            b += self.tau * f_mat[:, j + 1]
            b[0] = 0.0
            b[self.N] += mu * self.nu * self.h * self.p_fixed
            x[:, j + 1] = self._solve(b)

        return x


class AdjointProblem(PDEProblem):
    def __init__(
        self,
        a2: float,
        nu: float,
        l: float,
        T: float,
        N: int,
        M: int,
        y_target: NDArray[np.float64],
    ) -> None:
        super().__init__(a2, nu, l, T, N, M)
        self.y_target = y_target

    def solve(self, x_all: NDArray[np.float64]) -> NDArray[np.float64]:
        psi = np.zeros_like(x_all)
        psi[:, self.M] = 2.0 * (x_all[:, self.M] - self.y_target)

        for j in reversed(range(self.M)):
            b = psi[:, j + 1].copy()
            b[0] = 0.0
            psi[:, j] = self._solve(b)

        return psi
