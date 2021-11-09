# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:05:57 2021

# Ejemplos

@author: Arath Reyes
"""

from Programa_V2 import Derivative, VanillaEuropeanOption

"""
Valuación de un Put Americano (ej. 2 sesión 8)
"""

# miDerivadoAmericano = Derivative(60,60, 0.35, 0.06, 2, 1, "American", put = True)
# miDerivadoAmericano.summary()
# miDerivadoAmericano.compute()
# miDerivadoAmericano.summary()

S0 = 150
K = 140
r = 0.04
T = 7/12
N = 10
u = 1.33
d = 0.8666
vol = 0.15
M = 100

D = Derivative(S0, K, vol, r, N, T, kind = "Digital", M = M)
D.compute()
D.summary()
