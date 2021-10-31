# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:03:14 2021

@author: Arath Reyes
"""
import pandas as pd
import yfinance as yf
from numpy import maximum, exp, arange, zeros, mean, repeat
import numpy as np

"""
Aún falta el que la Americana de las alphas/deltas, y que toda la 
construcción de la asiática. Para ambos casos se necesita calcular el valor del 
subyacente en todos los nodos y los valores del derivado, para estas opciones
que dependen de la trayectoria se me ocurre hacer un árbol binario, pero
puede que sea más complicado :s

"""
###############################
####### Arbol Binario #########
###############################
class Node:
    def __init__(self, subyacente, derivado = None, suma_acumulada = None):
        self.subyacente = 0.
        self.derivado = derivado
        self.suma_acumulada = suma_acumulada # Yn
        self.u = 0.0
        self.d = 0.0
        
################################


def VanillaEuropeanOption(K, T, S0, r, N, u, d, put = False, hedge = False):
    # Declaración de variables
    delta = T/N
    q = (exp(r*delta)-d)/(u-d)
    B = exp(-r*delta)
    
    # Precios al tiempo N
    S = S0*(u**(arange(N,-1,-1)))*(d**(arange(0,N+1,1)))
    
    # Valor del payoff
    C = maximum((S-K)* ((-1)**put), zeros(N+1))
    
    # Valor del derivado tomando el árbol hacia atrás
    if hedge:
        delta = []
        for i in arange(N,0,-1):
            delta.append((C[0:i]-C[1:i+1])/(S[0:i]-S[1:i+1]))
            C = B*(q*C[0:i] + (1-q)*C[1:i+1])
            i-=1
            S = S0*(d**(arange(0, i+1, 1)))*(u**(arange(i,-1,-1)))
        return C[0], delta
    else:
        for i in arange(N,0,-1):
            C = B*((1-q)*C[1:i+1] + q*C[0:i])        
        return C[0]
        

def VanillaAmericanOption(K, T, S0, r, N, u, d, put = False):
    # Declaración de variables
    delta = T/N
    B = exp(-r*delta)
    q = (B**(-1) - d)/(u-d)
    
    # Precios al tiempo N
    S = S0*(d**(arange(N,-1,-1)))*(u**(arange(0,N+1,1)))
    
    # Valor del payoff
    C = maximum((S-K)* ((-1)**put),0)
    
    # Valor del derivado tomando el árbol hacia atrás
    for i in arange(N-1, -1,-1):
        S = S0*(d**(arange(i,-1,-1)))*(u**(arange(0,i+1,1)))
        C[:i+1] = B*(q*C[1:i+2] + (1-q)*C[0:i+1])
        C = C[:-1]
        C = maximum((S-K)* ((-1)**put),C)
    return C[0]

def AsianOption(K, T, S0, N, r, u, d):
    S = [S0*(d**(arange(0, i+1, 1)))*(u**(arange(i,-1,-1))) for i in arange(N, -1,-1)]
    """
    De momento esto sólo calcula los valores del subyacente, pero falta calcular Y y hacer la recursión
    hacia atrás
    """
    # for i in range(2**N):
        
    return

def Forward(K, S0, r):
    # Considerando que la tasa r está dada para todo el periodo de valuación,
    # es decir, exp(-r) = B(0,T)
    disc = exp(-r)
    F = S0 - disc*K
    return F


def UpNOutOption(K, T, S0, B, r, N, u, d, put = False):
    # Declaración de variables
    delta = T/N
    disc = exp(-r*delta)
    q = (disc**(-1) - d)/(u-d)
    
    # Valor del subyacente en N
    S = S0*(d**(arange(N,-1,-1)))*(u**(arange(0,N+1,1)))
    
    # Payoff
    C = maximum((S-K)* ((-1)**put),0)
    C[S>=B] = 0
    
    # Valor del derivado con hacia atrás
    for i in arange(N-1, -1,-1):
        S = S0*(d**(arange(i,-1,-1)))*(u**(arange(0,i+1,1)))
        C[:i+1] = disc*(q*C[1:i+2] + (1-q)*C[0:i+1])
        C = C[:-1]
        C[S>=B] = 0
    return C[0]

def UpNInOption(K, T, S0, B, r, N, u, d, put = False):
    """
    Notemos que:
        C_{call} = C_{call, uNo} + C_{call, uNi}
                        y
        C_{put} = C_{put, uNo} + C_{put, uNo}
    """
    C =  VanillaEuropeanOption(K, T, S0, r, N, u, d, put) - UpNOutOption(K, T, S0, B, r, N, u, d, put)
    return C


def LookBackOption():
    
    return

def DigitalOption():
    
    return


class Derivative:
    
    def __init__(self):
        self.strike = 0.0
        self.payoff = 0.0
        self.volatility = 0.0
        self.rate = 0.0
        self.periods = 0.0
        self.date = 0.0
        self.delta = []
        self.S0 = 0.0
        self.q = 0.0
        self.price = 0.0
        
    def compute(self, kind = "European", Call = True, *args):
        
        return

    # def plot(self):
        
    #     return
    
    def summary(self):
        
        return

