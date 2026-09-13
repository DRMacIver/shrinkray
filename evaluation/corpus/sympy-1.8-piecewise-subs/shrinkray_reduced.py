from sympy import *

a, b = symbols("0 b", real=1)
exp(sinh(Piecewise(a > b) / a)).subs({1: 1.0})
