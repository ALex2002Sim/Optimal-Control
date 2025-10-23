import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Literal, Tuple, List, Dict
from boundary import ForwardProblem, AdjointProblem


class ProjectionGradientSolver:
    def __init__(
        self,
        forward_problem: ForwardProblem,
        adjoint_problem: AdjointProblem,
        R: float,
        phi: NDArray[np.float64],
    ) -> None:
        self.forward = forward_problem
        self.adjoint = adjoint_problem
        self.R = R
        self.phi = phi
        self.h = forward_problem.h
        self.tau = forward_problem.tau

    def _project_L2_ball(self, f_mat: NDArray[np.float64]) -> NDArray[np.float64]:
        norm_sq = np.sum(f_mat**2) * self.h * self.tau
        norm = np.sqrt(norm_sq)
        if norm <= self.R:
            return f_mat
        return f_mat * (self.R / norm)

    def _compute_J(self, x_all: NDArray[np.float64]) -> float:
        diff = x_all[:, -1] - self.adjoint.y_target
        return np.sum(diff**2) * self.h

    def solve(
        self,
        f_init: NDArray[np.float64],
        max_iters: int = 50,
        tol: float = 1e-3,
        alpha_strategy: Literal["backtracking", "lipschitz_est"] = "backtracking",
        alpha0: float = 1e-2,
    ) -> Tuple[NDArray[np.float64], Dict[str, List[float]]]:
        f_current = f_init.copy()
        history = {"J": [], "norm_grad": [], "alpha": []}

        plt.ion()
        fig, (ax_temp, ax_J) = plt.subplots(1, 2, figsize=(18, 6))
        s = self.forward.s
        y_target = self.adjoint.y_target

        (line_xT,) = ax_temp.plot([], [], label="$x(s,T)$", color="cyan")
        ax_temp.plot(s, y_target, "--", label="$y(s)$", color="fuchsia")
        ax_temp.set_xlabel("$s$")
        ax_temp.set_ylabel("Temperature")
        ax_temp.set_title("Approximate distribution")
        ax_temp.legend()
        ax_temp.grid()

        (line_J,) = ax_J.semilogy([], [], label="$J(u)$", color="purple")
        ax_J.set_xlabel("Iteration")
        ax_J.set_ylabel("$J$")
        ax_J.set_title("Convergence of $J$")
        ax_J.legend()
        ax_J.grid()

        plt.show(block=False)

        for k in range(max_iters):
            x_all = self.forward.solve(f_current)
            J_curr = self._compute_J(x_all)
            psi = self.adjoint.solve(x_all)
            grad = psi.copy()
            grad_norm = np.sqrt(np.sum(grad**2) * self.h * self.tau)

            history["J"].append(J_curr)
            history["norm_grad"].append(grad_norm)

            print(f"iter {k:3d}: J={J_curr:.6e}, ||grad||={grad_norm:.6e}")

            if grad_norm < tol:
                print("→ converged")
                break

            alpha_used = alpha0

            if alpha_strategy == "backtracking":
                alpha = alpha0
                while True:
                    f_trial = f_current - alpha * grad
                    f_trial = self._project_L2_ball(f_trial)
                    x_trial = self.forward.solve(f_trial)
                    J_trial = self._compute_J(x_trial)
                    if J_trial < J_curr or alpha < 1e-12:
                        f_current = f_trial
                        alpha_used = alpha
                        break
                    alpha *= 0.5

            elif alpha_strategy == "lipschitz_est":
                eps = 1e-3
                d = -grad
                d_norm = np.sqrt(np.sum(d**2) * self.h * self.tau)
                if d_norm < 1e-12:
                    alpha = alpha0
                else:
                    d_unit = d / d_norm
                    f_pert = f_current + eps * d_unit
                    x_pert = self.forward.solve(f_pert)
                    psi_pert = self.adjoint.solve(x_pert)
                    L_est = (
                        np.sqrt(np.sum((psi_pert - grad) ** 2) * self.h * self.tau)
                        / eps
                    )
                    alpha = 1.0 / (L_est + 1e-6)
                alpha_used = min(alpha, 1.0)
                f_next = f_current - alpha_used * grad
                f_next = self._project_L2_ball(f_next)
                f_current = f_next
            else:
                raise ValueError("Unknown alpha strategy")

            history["alpha"].append(alpha_used)

            line_xT.set_data(s, x_all[:, -1])
            ax_temp.relim()
            ax_temp.autoscale_view()
            # ax_temp.set_title(
            #     f"Iteration {k} | $J$={J_curr:.3e} | $\\alpha$={alpha_used:.2e} ({alpha_strategy})"
            # )
            ax_temp.set_title(f"Iteration {k} | $J$={J_curr:.3e}")

            line_J.set_data(np.arange(1, len(history["J"]) + 1), history["J"])
            ax_J.relim()
            ax_J.autoscale_view()

            plt.pause(0.05)

        plt.ioff()
        plt.show()
        return f_current, history
