from sympy import *
from sympy.core.cache import clear_cache
x, y, z = symbols("x y z", real=True)
clear_cache()
expr = exp(sinh(Piecewise((x, y > x), (y, True)) / z))
expr.subs({1: 1.0})
