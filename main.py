import numpy as np
import matplotlib.pyplot as plt


# формула трапеций
def quad(f, N, h):
    integral = 0
    for i in range(N - 1):
        integral += (f[i] + f[i + 1]) / 2 * h

    return integral


# двумерная формула трапеций
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


# метод наискорейшего спуска
def fastest_descent(l, N, T, Nt, y, f, phi0, phi1, A, R, ar_y, x, psi):
    # используем метод золотго сечения для одномерной оптимизации шага
    eps = 0.01
    a = 0
    b = 1
    t = 0.618

    lambd = a + (1 - t) * (b - a)
    mu = a + t * (b - a)

    quad1 = quad_2d((f + lambd * psi) ** 2, N + 1, l / N, Nt + 1, T / Nt)
    if quad1 <= R**2:
        f1 = f + lambd * psi
    else:
        f1 = R * (f + lambd * psi) / np.sqrt(quad1)

    quad2 = quad_2d((f + mu * psi) ** 2, N + 1, l / N, Nt + 1, T / Nt)
    if quad2 <= R**2:
        f2 = f + mu * psi
    else:
        f2 = R * (f + mu * psi) / np.sqrt(quad2)

    x1 = original_task(l, N, T, Nt, f1, phi0, phi1, p, A)[-1, :]
    x2 = original_task(l, N, T, Nt, f2, phi0, phi1, p, A)[-1, :]

    J1 = quad((x1 - ar_y) ** 2, N + 1, l / N)
    J2 = quad((x2 - ar_y) ** 2, N + 1, l / N)

    for i in range(100):

        if (b - a) < eps:
            break

        if J1 > J2:
            a = lambd
            lambd = mu
            J1 = J2

            mu = a + t * (b - a)

            quad2 = quad_2d((f + mu * psi) ** 2, N + 1, l / N, Nt + 1, T / Nt)
            if quad2 <= R**2:
                f2 = f + mu * psi
            else:
                f2 = R * (f + mu * psi) / np.sqrt(quad2)

            x2 = original_task(l, N, T, Nt, f2, phi0, phi1, p, A)[-1, :]
            J2 = quad((x2 - ar_y) ** 2, N + 1, l / N)

        else:
            b = mu
            mu = lambd
            J2 = J1

            lambd = a + (1 - t) * (b - a)

            quad1 = quad_2d((f + lambd * psi) ** 2, N + 1, l / N, Nt + 1, T / Nt)
            if quad1 <= R**2:
                f1 = f + lambd * psi
            else:
                f1 = R * (f + lambd * psi) / np.sqrt(quad1)

            x1 = original_task(l, N, T, Nt, f1, phi0, phi1, p, A)[-1, :]
            J1 = quad((x1 - ar_y) ** 2, N + 1, l / N)

        return (a + b) / 2


# выбор шага alpha из условия монотонности градиента
def gradient_monotonicity(l, N, T, Nt, y, f, phi0, phi1, A, R, ar_y, x, psi):
    alpha = 1
    J = quad((x[-1, :] - ar_y) ** 2, N + 1, l / N)

    for i in range(100):

        quad1 = quad_2d((f + alpha * psi) ** 2, N + 1, l / N, Nt + 1, T / Nt)
        if quad1 <= R**2:
            f1 = f + alpha * psi
        else:
            f1 = R * (f + alpha * psi) / np.sqrt(quad1)

        x1 = original_task(l, N, T, Nt, f1, phi0, phi1, p, A)
        J1 = quad((x1[-1, :] - ar_y) ** 2, N + 1, l / N)

        if J1 < J:
            break
        else:
            alpha = alpha * 0.7

        if alpha < 0.001:
            break

    return alpha


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


def optimal_management1(l, N, T, Nt, y, phi0, phi1, a, R):
    f = R * R / l / T * np.ones((Nt + 1, N + 1))

    x_num = np.linspace(0, l, N + 1)
    x_NUM, t_NUM = np.meshgrid(x_num, np.linspace(0, T, Nt + 1))

    ar_y = np.zeros(N + 1)
    for i in range(N + 1):
        ar_y[i] = y(i * l / N)

    x_prev = np.zeros((Nt + 1, N + 1))
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)

    # изображаем желаемое распределение
    ax.plot(x_num, ar_y, label="Заданное")
    for step in range(1, 2000):

        f_prev = f.copy()

        x = original_task(l, N, T, Nt, f_prev, phi0, phi1, p, a)
        psi = conjugate_task(l, N, T, Nt, x[-1, :], y, a)

        alpha = fastest_descent(l, N, T, Nt, y, f_prev, phi0, phi1, a, R, ar_y, x, psi)
        # print(alpha)

        # пересчет управления f
        quad_ = quad_2d((f_prev + alpha * psi) ** 2, N + 1, l / N, Nt + 1, T / Nt)
        if quad_ <= R**2:
            f = f_prev + alpha * psi
        else:
            f = R * (f_prev + alpha * psi) / np.sqrt(quad_)

        if step % 1 == 0:
            # изображаем айденное распределение
            ax.plot(
                x_num, x[-1, :], linestyle="--", label="Приближенное, {}".format(step)
            )

        print(
            step,
            abs(
                quad((x_prev[-1, :] - ar_y) ** 2, N + 1, l / N)
                - quad((x[-1, :] - ar_y) ** 2, N + 1, l / N)
            ),
        )

        # Критерий окончания 1
        if np.sqrt(quad_2d((f_prev - f) ** 2, N + 1, l / N, Nt + 1, T / Nt)) < 0.0001:
            print("Критерий 1")
            break

        # Критерий окончания 2
        if (
            abs(
                quad((x_prev[-1, :] - ar_y) ** 2, N + 1, l / N)
                - quad((x[-1, :] - ar_y) ** 2, N + 1, l / N)
            )
            < 0.00001
        ):
            print("Критерий 2")
            break

        # Критерий окончания 3
        if np.sqrt(quad_2d(psi**2, N + 1, N / l, Nt + 1, T / Nt)) < 0.0001:
            print("Критерий 3")
            break

        x_prev = x.copy()
    ax.set_title("Распределения, R= {}".format(R))
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


# def optimal_management2(l, N, T, Nt, y, phi0, phi1, a, R):
#     f = R * R / l / T * np.ones((Nt + 1, N + 1))
#
#     x_num = np.linspace(0, l, N + 1)
#     x_NUM, t_NUM = np.meshgrid(x_num, np.linspace(0, T, Nt + 1))
#
#     ar_y = np.zeros(N + 1)
#     for i in range(N + 1):
#         ar_y[i] = y(i * l / N)
#
#     x_prev = np.zeros((Nt + 1, N + 1))
#
#     for step in range(1, 2000):
#
#         f_prev = f.copy()
#
#         x = original_task(l, N, T, Nt, f_prev, phi0, phi1, p, a)
#         psi = conjugate_task(l, N, T, Nt, x[-1, :], y, a)
#
#         alpha = gradient_monotonicity(l, N, T, Nt, y, f_prev, phi0, phi1, a, R, ar_y, x, psi)
#         # print(alpha)
#
#         # пересчет управления f
#         quad_ = quad_2d((f_prev + alpha * psi) ** 2, N + 1, l / N, Nt + 1, T / Nt)
#         if quad_ <= R ** 2:
#             f = f_prev + alpha * psi
#         else:
#             f = R * (f_prev + alpha * psi) / np.sqrt(quad_)
#
#         if step % 1 == 0:
#             fig = plt.figure()
#             ax = fig.add_subplot(1, 2, 1)
#
#             # изображаем желаемое распределение и найденное
#             ax.plot(x_num, ar_y, label="Заданное")
#             ax.plot(x_num, x[-1, :], label="Приближенное, {}".format(step))
#             ax.set_title("Распределения")
#             ax.legend()
#             ax.grid(True)
#
#             axes = fig.add_subplot(1, 2, 2, projection="3d")
#             axes.plot_surface(x_NUM, t_NUM, f, cmap='plasma')
#             axes.set_title("Управление f")
#             axes.set_xlabel("s")
#             axes.set_ylabel("t")
#
#             fig.set_figheight(6)
#             fig.set_figwidth(12)
#             plt.show()
#
#         print(abs(quad((x_prev[-1, :] - ar_y) ** 2, N + 1, l / N) -
#                   quad((x[-1, :] - ar_y) ** 2, N + 1, l / N)))
#
#         # Критерий окончания 1
#         if np.sqrt(quad_2d((f_prev - f) ** 2, N + 1, l / N, Nt + 1, T / Nt)) < 0.0001:
#             print("Критерий 1")
#             break
#
#         # Критерий окончания 2
#         if abs(quad((x_prev[-1, :] - ar_y) ** 2, N + 1, l / N) -
#                quad((x[-1, :] - ar_y) ** 2, N + 1, l / N)) < 0.001:
#             print("Критерий 2")
#             break
#
#         # Критерий окончания 3
#         if np.sqrt(quad_2d(psi ** 2, N + 1, N / l, Nt + 1, T / Nt)) < 0.0001:
#             print("Критерий 3")
#             break
#
#         x_prev = x.copy()


def optimal_management5(l, N, T, Nt, y, phi0, phi1, a, R):

    f = R * R / l / T * np.ones((Nt + 1, N + 1))

    x_num = np.linspace(0, l, N + 1)
    x_NUM, t_NUM = np.meshgrid(x_num, np.linspace(0, T, Nt + 1))

    ar_y = np.zeros(N + 1)
    for i in range(N + 1):
        ar_y[i] = y(i * l / N)

    x_prev = np.zeros((Nt + 1, N + 1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)

    # изображаем желаемое распределение
    ax.plot(x_num, ar_y, label="Заданное")

    for step in range(1, 2000 + 1):

        f_prev = f.copy()

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
            # изображаем найденное распределение
            ax.plot(
                x_num, x[-1, :], linestyle="--", label="Приближенное, {}".format(step)
            )

        print(
            step,
            abs(
                quad((x_prev[-1, :] - ar_y) ** 2, N + 1, l / N)
                - quad((x[-1, :] - ar_y) ** 2, N + 1, l / N)
            ),
        )

        # Критерий окончания 1
        if np.sqrt(quad_2d((f_prev - f) ** 2, N + 1, l / N, Nt + 1, T / Nt)) < 0.0001:
            print("Критерий 1")
            break

        # Критерий окончания 2
        if (
            abs(
                quad((x_prev[-1, :] - ar_y) ** 2, N + 1, l / N)
                - quad((x[-1, :] - ar_y) ** 2, N + 1, l / N)
            )
            < 0.001
        ):
            print("Критерий 2")
            break

        # Критерий окончания 3
        if np.sqrt(quad_2d(psi**2, N + 1, N / l, Nt + 1, T / Nt)) < 0.0001:
            print("Критерий 3")
            break

    ax.set_title("Распределения, R= {}".format(R))
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

test = 1

if test == 1:

    def y(s):
        return (s - l) ** 2 + T**3  # + 2*np.cos(3*s)

    def phi0(s):
        return (s - l) ** 2

    def phi1(s):
        return 0

    def p(t):
        return -2 * l

elif test == 2:

    def y(s):
        return np.arctan(5 * (s - 0.5))

    def phi0(s):
        return np.arctan(5 * (-0.5))

    def phi1(s):
        return 5 * (1 + (5 * (s - 0.5)) ** 2) ** (-1)

    def p(t):
        return t**3


R = 10

optimal_management5(l, N, T, Nt, y, phi0, phi1, a, R)
