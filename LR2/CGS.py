# CGS.py
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, List
from boundary import WaveSolver


class ConditionalGradientSolver:
    """
    Реализация условного градиента (Frank–Wolfe) для управления p(t) (левый Neumann)
    Вариант 7: метод условного градиента, управление p(t).
    Способы выбора alpha: 'line_search' (номер 1 в методичке) и 'apriori' (номер 5).
    """

    def __init__(
        self,
        wave: WaveSolver,
        R0: float,
        phi0: NDArray[np.float64],
        phi1: NDArray[np.float64],
        y_target: NDArray[np.float64],
        f_mat: NDArray[np.float64] = None,
    ) -> None:
        self.wave = wave
        self.R0 = R0
        self.phi0 = phi0
        self.phi1 = phi1
        self.y_target = y_target
        self.f_mat = f_mat if f_mat is not None else np.zeros((wave.N + 1, wave.M + 1))
        self.h = wave.h
        self.tau = wave.tau
        self.t = wave.t

    def _compute_J(self, x_all: NDArray[np.float64]) -> float:
        # функционал J(u) = \int |x(s,T)-y(s)|^2 ds
        diff = x_all[:, -1] - self.y_target
        return np.sum(diff ** 2) * self.h

    def _compute_bar_p(self, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Формула для условного градиента для p:
        \bar p_k(t) = - R0 * psi(0,t) / || psi(0,·) ||_L2
        (см. методичку, раздел для струны)
        """
        psi0 = psi[0, :]  # значение psi в левом конце для всех t
        norm = np.sqrt(np.trapz(psi0 ** 2, self.t))
        if norm < 1e-16:
            return np.zeros_like(psi0)
        return -self.R0 * psi0 / norm

    def _line_search_alpha(self, p_k: NDArray[np.float64], p_bar: NDArray[np.float64]) -> float:
        """
        Выбор alpha в [0,1] минимизирующий J(p_k + alpha*(p_bar - p_k)).
        Поскольку J требует запуска прямой задачи, делаем дискретный поиск на сетке.
        """
        alphas = np.linspace(0.0, 1.0, 21)
        best_a = 0.0
        best_J = np.inf
        for a in alphas:
            p_trial = p_k + a * (p_bar - p_k)
            x_trial = self.wave.forward_solve(p_trial, self.f_mat, self.phi0, self.phi1)
            Jt = self._compute_J(x_trial)
            if Jt < best_J:
                best_J = Jt
                best_a = a
        # можно сделать локальное уточнение вокруг best_a, но для простоты оставим так
        return best_a

    def solve(
        self,
        p0: NDArray[np.float64],
        max_iters: int = 200,
        tol: float = 1e-4,
        alpha_strategy: str = "line_search",
    ) -> Tuple[NDArray[np.float64], Dict[str, List[float]]]:
        p_current = p0.copy()
        history = {"J": [], "alpha": []}

        for k in range(max_iters):
            # прямая задача
            x_all = self.wave.forward_solve(p_current, self.f_mat, self.phi0, self.phi1)
            J_curr = self._compute_J(x_all)

            # сопряженная задача (обратный ход)
            psi = self.wave.adjoint_solve(x_all, self.y_target)

            # генерируем экстремальную точку
            p_bar = self._compute_bar_p(psi)

            # выбираем шаг alpha
            if alpha_strategy == "line_search":
                alpha = self._line_search_alpha(p_current, p_bar)
            elif alpha_strategy == "apriori":
                alpha = 1.0 / (k + 1.0)  # пример априорной последовательности; удовлетворяет сумме = inf, limit 0
            else:
                raise ValueError("Unknown alpha_strategy")

            # обновление
            p_next = p_current + alpha * (p_bar - p_current)

            # сохранение истории
            history["J"].append(J_curr)
            history["alpha"].append(alpha)

            print(f"iter {k:3d}: J={J_curr:.6e}, alpha={alpha:.4e}")

            # критерий останова
            if np.linalg.norm(p_next - p_current) * np.sqrt(self.tau) < tol:
                p_current = p_next
                print("→ converged by control change")
                break

            p_current = p_next

        return p_current, history
