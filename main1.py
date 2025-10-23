import numpy as np
import matplotlib.pyplot as plt


# формула трапеций
def quad(f, N, h):
    integral = 0
    for i in range(N - 1):
        integral += (f[i] + f[i + 1]) / 2 * h

    return integral


# исходная краевая задача
def original_task(l, N, T, Nt, f, phi, p, a, nu):
    h = l / N
    tau = T / Nt

    u = np.zeros(N + 1)
    up = np.zeros(N + 1)

    for i in range(N + 1):
        u[i] = phi(i * h)

    A = a * a / h / h
    C = 2 * A + 1 / tau

    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)

    for j in range(1, Nt + 1):

        up = u.copy()

        alpha[1] = 1 / h / (1 / h + h / 2 / tau / a / a)
        beta[1] = (h / 2 / a / a * f(0, j * tau) + h / 2 / tau / a / a * up[0]) / (
            1 / h + h / 2 / tau / a / a
        )

        for i in range(1, N):
            alpha[i + 1] = A / (C - A * alpha[i])
            beta[i + 1] = (f(i * h, j * tau) + up[i] / tau + A * beta[i]) / (
                C - A * alpha[i]
            )

        u[N] = (
            nu * p[j]
            + h / 2 / tau / a / a * up[N]
            + h / 2 / a / a * f(l, j * tau)
            + beta[N] / h
        ) / (1 / h + h / 2 / a / a / tau + nu - alpha[N] / h)

        for i in range(N - 1, -1, -1):
            u[i] = alpha[i + 1] * u[i + 1] + beta[i + 1]

    return u


# сопряженная краевая задача
def conjugate_task(l, N, T, Nt, x, y, a, nu):
    h = l / N
    tau = T / Nt

    u = np.zeros(N + 1)
    up = np.zeros(N + 1)

    psi = np.zeros(Nt + 1)

    for i in range(N + 1):
        u[i] = 2 * (x[i] - y(i * h))

    psi[Nt] = u[N]

    A = a * a / h / h
    C = 2 * A + 1 / tau

    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)

    # xnum = np.linspace(0, l, N+1)

    # plt.plot(xnum, x)
    # plt.show()

    for j in range(1, Nt + 1):

        up = u.copy()

        alpha[1] = 1 / h / (1 / h + h / 2 / tau / a / a)
        beta[1] = h / 2 / tau / a / a * up[0] / (1 / h + h / 2 / tau / a / a)

        for i in range(1, N):
            alpha[i + 1] = A / (C - A * alpha[i])
            beta[i + 1] = (up[i] / tau + A * beta[i]) / (C - A * alpha[i])

        u[N] = (h / 2 / tau / a / a * up[N] + beta[N] / h) / (
            1 / h + h / 2 / a / a / tau + nu - alpha[N] / h
        )

        for i in range(N - 1, -1, -1):
            u[i] = alpha[i + 1] * u[i + 1] + beta[i + 1]

        psi[Nt - j] = u[N]

    return psi


def p_conditions(psi, pmin, pmax, Nt):
    p_ = np.zeros(Nt + 1)

    for i in range(Nt + 1):

        if psi[i] >= 0:
            p_[i] = pmin
        else:
            p_[i] = pmax
    return p_


# априорное задание шага (вариант 5)
def optimal_management5(l, N, T, Nt, y, pmin, pmax, f, phi, a, nu):
    p = (pmin + pmax) * np.ones(Nt + 1)

    xnum = np.linspace(0, l, N + 1)
    tnum = np.linspace(0, T, Nt + 1)

    ar_y = np.zeros(N + 1)
    for i in range(N + 1):
        ar_y[i] = y(i * l / N)

    x_prev = np.zeros(N + 1)

    fig, ax = plt.subplots(1, 2)

    # изображаем желаемое распределение и найденное
    ax[0].plot(xnum, ar_y, label="Заданное")

    for step in range(1, 1000):

        x = original_task(l, N, T, Nt, f, phi, p, a, nu)
        psi = conjugate_task(l, N, T, Nt, x, y, a, nu)

        if step % 100 == 0:

            ax[0].plot(xnum, x, linestyle="--", label="Приближенное, {}".format(step))

        p_ = p_conditions(psi, pmin, pmax, Nt)

        # пересчет управления p
        dp = 1 / step * (p_ - p)

        # критерий окончаня 1
        if np.sqrt(quad(dp**2, Nt + 1, T / Nt)) < 0.0001:
            break

        # критерий окончания 2
        if (
            abs(
                quad((x_prev - ar_y) ** 2, N + 1, l / N)
                - quad((x - ar_y) ** 2, N + 1, l / N)
            )
            < 0.001
        ):
            break
        print(
            abs(
                quad((x_prev - ar_y) ** 2, N + 1, l / N)
                - quad((x - ar_y) ** 2, N + 1, l / N)
            )
        )

        # критерий окончания 3
        if np.sqrt(abs(quad(a * a * nu * psi, Nt + 1, T / Nt))) < 0.01:
            break

        p = p + dp

        x_prev = x.copy()
    # изображаем управление p
    ax[1].plot(tnum, p)

    ax[0].set_title("Распределения")
    ax[1].set_title("Управление p")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    plt.show()


# задание шага по условию липшица
def optimal_management3(l, N, T, Nt, y, pmin, pmax, f, phi, a, nu):
    def p_conditions(psi, pmin, pmax, Nt):

        p_ = np.zeros(Nt + 1)

        for i in range(Nt + 1):

            if psi[i] >= 0:
                p_[i] = pmin
            else:
                p_[i] = pmax
        return p_

    xnum = np.linspace(0, l, N + 1)
    tnum = np.linspace(0, T, Nt + 1)

    ar_y = np.zeros(N + 1)
    for i in range(N + 1):
        ar_y[i] = y(i * l / N)

    p = (pmin + pmax) * np.ones(Nt + 1)
    x_prev = np.zeros(N + 1)

    # вычисление rho через константу Липшица
    e0 = 1
    e = 0.01

    c0 = max(a * a * nu / e0, 1 / a / a / e0)
    c1 = max((a**4 * nu * nu + 2 * l) / a / a / nu, 2 * l * l / a / a)

    L = np.sqrt(2 * c0 * c1)

    rho = 2 / (L + 2 * e)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(xnum, ar_y, label="Заданное")
    ax[0].set_title("Распределения")
    for step in range(1, 1000):

        x = original_task(l, N, T, Nt, f, phi, p, a, nu)
        psi = conjugate_task(l, N, T, Nt, x, y, a, nu)

        if step % 10 == 0:

            # изображаем желаемое распределение и найденное

            ax[0].plot(xnum, x, linestyle="--", label="Приближенное, {}".format(step))

            # изображаем управление p

        p_ = p_conditions(psi, pmin, pmax, Nt)

        # расчет коэффициента alpha
        alpha = min(
            1,
            rho
            * abs(quad(a * a * nu * psi * (p_ - p), Nt + 1, T / Nt))
            / quad((p_ - p) ** 2, Nt + 1, T / Nt),
        )

        # пересчет управления p
        dp = alpha * (p_ - p)

        # критерий окончаня 1
        if np.sqrt(quad(dp**2, Nt + 1, T / Nt)) < 0.0001:
            break

        # критерий окончания 2
        if (
            abs(
                quad((x_prev - ar_y) ** 2, N + 1, l / N)
                - quad((x - ar_y) ** 2, N + 1, l / N)
            )
            < 0.001
        ):
            break
        print(
            abs(
                quad((x_prev - ar_y) ** 2, N + 1, l / N)
                - quad((x - ar_y) ** 2, N + 1, l / N)
            )
        )

        # критерий окончания 3
        if np.sqrt(abs(quad(a * a * nu * psi, Nt + 1, T / Nt))) < 0.01:
            break

        p = p + dp

        x_prev = x.copy()
    ax[0].legend()
    ax[0].grid(True)

    fig.set_figheight(6)
    fig.set_figwidth(12)
    ax[1].plot(tnum, p)
    ax[1].set_title("Управление p")
    ax[1].grid(True)
    plt.show()


def optimal_management2(l, N, T, Nt, y, pmin, pmax, f, phi, a, nu):
    def p_conditions(psi, pmin, pmax, Nt):
        p_ = np.zeros(Nt + 1)
        for i in range(Nt + 1):
            if psi[i] >= 0:
                p_[i] = pmin
            else:
                p_[i] = pmax
        return p_

    xnum = np.linspace(0, l, N + 1)
    tnum = np.linspace(0, T, Nt + 1)

    ar_y = np.zeros(N + 1)
    for i in range(N + 1):
        ar_y[i] = y(i * l / N)

    p = (pmin + pmax) * np.ones(Nt + 1)
    x_prev = np.zeros(N + 1)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(xnum, ar_y, label="Заданное")
    ax[0].set_title("Распределения")

    for step in range(1, 12):

        x = original_task(l, N, T, Nt, f, phi, p, a, nu)
        psi = conjugate_task(l, N, T, Nt, x, y, a, nu)

        if step % 1 == 0:

            ax[0].plot(xnum, x, linestyle="--", label="Приближенное, {}".format(step))

        p_ = p_conditions(psi, pmin, pmax, Nt)
        # Условие 2: выбираем шаг методом дробления
        alpha = 1  # Начальное значение шага
        J_prev = quad((x_prev - ar_y) ** 2, N + 1, l / N)
        while True:
            dp = alpha * (p_ - p)
            p_test = p + dp
            x_test = original_task(l, N, T, Nt, f, phi, p_test, a, nu)
            J_test = quad((x_test - ar_y) ** 2, N + 1, l / N)
            if J_test < J_prev:  # Проверка монотонности
                break
            alpha *= 0.5  # Уменьшаем шаг
            print("alpha = ", alpha)

        # Пересчёт управления p
        dp = alpha * (p_ - p)
        p = p + dp

        # критерий окончаня 1
        if np.sqrt(quad(dp**2, Nt + 1, T / Nt)) < 0.0001:
            break

        # критерий окончания 2
        if (
            abs(
                quad((x_prev - ar_y) ** 2, N + 1, l / N)
                - quad((x - ar_y) ** 2, N + 1, l / N)
            )
            < 0.001
        ):
            break

        print(
            step,
            abs(
                quad((x_prev - ar_y) ** 2, N + 1, l / N)
                - quad((x - ar_y) ** 2, N + 1, l / N)
            ),
        )

        # критерий окончания 3
        if np.sqrt(abs(quad(a * a * nu * psi, Nt + 1, T / Nt))) < 0.01:
            break

        x_prev = x.copy()

    ax[0].legend()
    ax[0].grid(True)

    fig.set_figheight(6)
    fig.set_figwidth(12)
    ax[1].plot(tnum, p)
    ax[1].set_title("Управление p")
    ax[1].grid(True)
    plt.show()


# тестовые данные

l = 1
N = 100

T = 10
Nt = 100

a = 5
nu = 19


test = 4

if test == 1:

    def phi(s):
        return 0

    def f(s, t):
        return 2 * s * s * t - 2 * a * a * t * t

    def y(s):
        return s * s * T * T

elif test == 2:

    def y(s):
        return np.cos(s * T)

    # Правая часть
    def f(s, t):
        return (a**2 * t**2 - s) * np.cos(s * t)

    # Начальные условия
    def phi(s):
        return 1

elif test == 3:

    def phi(s):
        return 0

    def f(s, t):
        return -2 * a**2 * t**2

    def y(s):
        return s**2 * T**2

elif test == 4:

    def phi(s):
        return 0

    def f(s, t):
        return s**2 - 2 * a**2 * t

    def y(s):
        return s**2 * T

elif test == 5:

    def phi(s):
        return 5

    def f(s, t):
        return 0

    def y(s):
        return 5


pmin = -10000
pmax = 10000

# optimal_management5(l, N, T, Nt, y, pmin, pmax, f, phi, a, nu)

optimal_management2(l, N, T, Nt, y, pmin, pmax, f, phi, a, nu)
