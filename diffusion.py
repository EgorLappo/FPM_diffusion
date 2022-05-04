from scipy.integrate import quad
from math import exp

def a_WF_selection(N, s, h):
    def a(x):
        return

    return a


def b(x):
    return x*(1-x)


def P0(p, a, b):

    def integrand(x):
        if b(x) == 0:
            return 0
        else:
            return a(x)/b(x)

    def psi(y):
        if y == 1.0:
            return 0
        else:
            power, _ = quad(integrand, 0, y)
            return exp(-2*power)

    numerator, _ = quad(psi, p, 1)
    denominator, _ = quad(psi, 0, 1)

    return numerator/denominator


def P1(p, a, b):

    def integrand(x):
        if b(x) == 0:
            return 0
        else:
            return a(x)/b(x)

    def psi(y):
        if y == 1.0:
            return 0
        else:
            power, _ = quad(integrand, 0, y)
        return exp(-2*power)

    numerator, _ = quad(psi, 0, p)
    denominator, _ = quad(psi, 0, 1)

    return numerator/denominator


def t_bar(p, a, b):

    def integrand(x):
        if b(x) == 0:
            return 0
        else:
            return a(x)/b(x)

    def psi(y):
        if y == 1.0:
            return 0
        else:
            power, _ = quad(integrand, 0, y)
            return exp(-2*power)

    def t(x):

        if (x <= p):
            numerator, _ = quad(psi, p, 1)
            denominator, _ = quad(psi, 0, 1)
            P = numerator/denominator

            psi_int, _ = quad(psi, 0, x)

            if psi(x)*b(x) == 0.0:
                ans = 0
            else:
                ans = 2*P*psi_int/(b(x)*psi(x))

        else:
            numerator, _ = quad(psi, 0, p)
            denominator, _ = quad(psi, 0, 1)
            P = numerator/denominator

            psi_int, _ = quad(psi, x, 1)

            if psi(x)*b(x) == 0.0:
                ans = 0
            else:
                ans = 2*P*psi_int/(b(x)*psi(x))

        return ans

    return quad(t, 0, 1)[0]
