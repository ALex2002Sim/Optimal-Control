import numpy as np
import matplotlib.pyplot as plt


# формула трапеций
def quad(f, N, h):
    integral = 0
    for i in range(N - 1):
        integral += (f[i] + f[i + 1]) / 2 * h

    return integral


def quad_2d(f, N, h, Nt, tau):
    integral = 0

    for j in range(Nt):

        if j == 0:
            integral += tau / 2 * quad(f[j, :], N, h)
        elif j == Nt - 1:
            integral += tau / 2 * quad(f[j, :], N, h)
        else:
            integral += tau * quad(f[j, :], N, h)

    return integral


# исходная краевая задача
def original_task(l, N, T, Nt, f, phi0, phi1, p, a):
    h = l / N
    tau = T / Nt

    u = np.zeros((Nt + 1, N + 1))

    for i in range(N + 1):
        u[0, i] = phi0(i * h)

    for i in range(N + 1):
        u[1, i] = (
            u[0, i]
            + tau * phi1(i * h)
            + tau
            * tau
            / 2
            * (
                a
                * a
                / h
                / h
                * (phi0((i - 1) * h) - 2 * phi0(i * h) + phi0((i + 1) * h))
                + f[1, i]
            )
        )

    A = a * a / h / h
    C = 2 * A + 1 / tau / tau

    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)

    for j in range(2, Nt + 1):

        alpha[1] = 1 / h / (1 / h + h / 2 / tau / tau / a / a)
        beta[1] = (
            -p(j * tau)
            + h
            / 2
            / a
            / a
            * (1 / tau / tau * (2 * u[j - 1, 0] - u[j - 2, 0]) + f[j, 0])
        ) / (1 / h + h / 2 / tau / tau / a / a)

        for i in range(1, N):
            alpha[i + 1] = A / (C - A * alpha[i])
            beta[i + 1] = (
                f[j, i]
                + 2 * u[j - 1, i] / tau / tau
                - u[j - 2, i] / tau / tau
                + A * beta[i]
            ) / (C - A * alpha[i])

        u[j, N] = (
            h / 2 / tau / tau / a / a * (2 * u[j - 1, N] - u[j - 2, N])
            + h / 2 / a / a * f[j, N]
            + beta[N] / h
        ) / (1 / h + h / 2 / a / a / tau / tau - alpha[N] / h)

        for i in range(N - 1, -1, -1):
            u[j, i] = alpha[i + 1] * u[j, i + 1] + beta[i + 1]

    return u


def conjugate_task(l, N, T, Nt, x, y, a):
    h = l / N
    tau = T / Nt

    u = np.zeros((Nt + 1, N + 1))

    for i in range(N + 1):
        u[-2, i] = -2 * tau * (x[i] - y(i * h))

    A = a * a / h / h
    C = 2 * A + 1 / tau / tau

    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)

    for j in range(2, Nt + 1):

        alpha[1] = 1 / h / (1 / h + h / 2 / tau / tau / a / a)
        beta[1] = (
            h
            / 2
            / a
            / a
            * (1 / tau / tau * (2 * u[Nt - (j - 1), 0] - u[Nt - (j - 2), 0]))
        ) / (1 / h + h / 2 / tau / tau / a / a)

        for i in range(1, N):
            alpha[i + 1] = A / (C - A * alpha[i])
            beta[i + 1] = (
                2 * u[Nt - (j - 1), i] / tau / tau
                - u[Nt - (j - 2), i] / tau / tau
                + A * beta[i]
            ) / (C - A * alpha[i])

        u[Nt - j, N] = (
            h / 2 / tau / tau / a / a * (2 * u[Nt - (j - 1), N] - u[Nt - (j - 2), N])
            + beta[N] / h
        ) / (1 / h + h / 2 / a / a / tau / tau - alpha[N] / h)

        for i in range(N - 1, -1, -1):
            u[Nt - j, i] = alpha[i + 1] * u[Nt - j, i + 1] + beta[i + 1]

    return u


def optimal_management5(l, N, T, Nt, y, f, phi0, phi1, a, R):
    f = R * R / l / T * np.ones((Nt + 1, N + 1))

    x_num = np.linspace(0, l, N + 1)
    x_NUM, t_NUM = np.meshgrid(x_num, np.linspace(0, T, Nt + 1))

    ar_y = np.zeros(N + 1)
    for i in range(N + 1):
        ar_y[i] = y(i * l / N)

    x_prev = np.zeros((Nt + 1, N + 1))
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)

    # изображаем желаемое распределение и найденное
    ax.plot(x_num, ar_y, label="Заданное")
    for step in range(1, 200 + 1):

        x = original_task(l, N, T, Nt, f, phi0, phi1, p, a)
        psi = conjugate_task(l, N, T, Nt, x[-1, :], y, a)

        alpha = 1 / step

        if quad_2d((f + alpha * psi) ** 2, N + 1, l / N, Nt + 1, T / Nt) <= R**2:

            f = f + alpha * psi

        else:

            f = (
                R
                * (f + alpha * psi)
                / np.sqrt(quad_2d((f + alpha * psi) ** 2, N + 1, l / N, Nt + 1, T / Nt))
            )

        if step % 20 == 0:

            ax.plot(x_num, x[-1, :], label="Приближенное, {}".format(step))

        print(
            step,
            abs(
                quad((x_prev[-1, :] - ar_y) ** 2, N + 1, l / N)
                - quad((x[-1, :] - ar_y) ** 2, N + 1, l / N)
            ),
        )
    ax.set_title("Распределения")
    ax.legend()
    ax.grid(True)

    axes = fig.add_subplot(1, 2, 2, projection="3d")
    axes.plot_surface(x_NUM, t_NUM, f, cmap="plasma")
    axes.set_title("Управление f")
    axes.set_xlabel("s")
    axes.set_ylabel("t")

    fig.set_figheight(6)
    fig.set_figwidth(12)
    plt.show()


a = 1
l = 1

T = 1

N = 100
Nt = 100


def y(s):
    return (s - l) ** 2 + T**3


def phi0(s):
    return (s - l) ** 2


def phi1(s):
    return 0


def p(t):
    return -2 * l


def f(s, t):
    return -2 * a * a + 6 * t


R = 1

optimal_management5(l, N, T, Nt, y, f, phi0, phi1, a, R)
# original_task(l, N, T, Nt, f, phi0, phi1, p, a)
