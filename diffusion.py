from scipy.integrate import quad
from math import exp

""" EXAMPLE DRIFT AND DIFFUSION FUNCTIONS """

def a_WF_selection(N, s, h):
    """Example drift function a(x) for WF model with selection.

    Reference: Ewens eq. 5.43.

    Args:
        N (int): population size.
        s (float): selection coefficient.
        h (float): dominance coefficient.

    Returns:
        func: drift function for the specified parameters.
    """
    alpha = 2*N*s

    def ans(x):
        return alpha * x * (1-x) * (x + h*(1-2*x))
    return ans

def a_conformity_3rm(N, s, D):
    """Example drift function a(x) for the FPM of conformist transmission with n=3 role models.

    Args:
        N (int): population size.
        s (float): selection coefficient.
        D (float): conformity coefficient.

    Returns:
        func: drift function for the specified parameters.
    """
    alpha = s*N
    Delta = D*N

    def ans(x):
        return x*(alpha - alpha*x - Delta*(-1 + x)*(-1 + 2*x))

    return ans

def a_conformity_5rm(N, s, D3, D4):
    """Example drift function a(x) for the FPM of conformist transmission with n=5 role models.

    Args:
        N (int): population size.
        s (float): selection coefficient.
        D3 (float): conformity coefficient D(3).
        D4 (float): conformity coefficient D(4).

    Returns:
        func: drift function for the specified parameters.
    """
    alpha = s*N
    Delta3 = D3*N
    Delta4 = D4*N

    def ans(x):
        return alpha*x - alpha*x**2 + (1 - x)*x*(-1 + 2*x)*(Delta4 - (2*Delta3 - Delta4)*(-1 + x)*x)

    return ans


def b_std(x):
    """Standard diffusion function used in most diffusion approximations.

    Reference: Ewens eq. 5.5
    """
    return x*(1-x)

""" DIFFUSION APPROXIMATION QUANTITIES """

def P0(p, a, b):
    """Probability of fixation to p=0.
    
    Reference: Ewens eq. 4.15

    Args:
        p (float): initial frequency.
        a (func): drift function.
        b (func): diffusion function.

    Returns:
        float: probability of fixation to p=0.
    """

    def integrand(x):
        """
        Integrand in Ewens eq. 4.16 
        """
        if b(x) == 0:
            return 0
        else:
            return a(x)/b(x)

    def psi(y):
        """ 
        Function defined in Ewens eq. 4.16
        """
        if y == 1.0:
            return 0
        else:
            power, _ = quad(integrand, 0, y)
            return exp(-2*power)

    numerator, _ = quad(psi, p, 1)
    denominator, _ = quad(psi, 0, 1)

    return numerator/denominator


def P1(p, a, b):
    """Probability of fixation to p=1.
    
    Reference: Ewens eq. 4.17

    Args:
        p (float): initial frequency.
        a (func): drift function.
        b (func): diffusion function.

    Returns:
        float: probability of fixation to p=0.
    """


    def integrand(x):
        """
        Integrand in Ewens eq. 4.16 
        """
        if b(x) == 0:
            return 0
        else:
            return a(x)/b(x)

    def psi(y):
        """ 
        Function defined in Ewens eq. 4.16
        """
        if y == 1.0:
            return 0
        else:
            power, _ = quad(integrand, 0, y)
        return exp(-2*power)

    numerator, _ = quad(psi, 0, p)
    denominator, _ = quad(psi, 0, 1)

    return numerator/denominator


def t_bar(p, a, b):
    """Time to absorption.

    Reference: Ewens eq. 4.21

    Args:
        p (float): initial frequency.
        a (_type_): drift function.
        b (_type_): diffusion function.

    Returns:
        float: time to absorption, in units of (twice) population size.
    """

    def integrand(x):
        """
        Integrand in Ewens eq. 4.16 
        """
        if b(x) == 0:
            return 0
        else:
            return a(x)/b(x)

    def psi(y):
        """ 
        Function defined in Ewens eq. 4.16
        """
        if y == 1.0:
            return 0
        else:
            power, _ = quad(integrand, 0, y)
            return exp(-2*power)

    def t(x):
        """ 
        Function defined in Ewens eqs. 4.22, 4.23 
        """

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
