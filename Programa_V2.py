# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:03:14 2021

@author: Caballero Salazar Adriana
         Cruz Estrada Valeria Lucero
         Cuéllar Chávez Eduardo de Jesús
         Paz Garduño Valeria Lizette
         Reyes López Arath Alejandro
"""
from numpy import maximum, exp, arange, zeros, mean, repeat
import numpy as np
import itertools
from toolz import partition
import math

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

def VanillaAmericanOption(K, T, S0, r, N, u, d, put = False, hedge = False):
        # Declaración de variables
    delta = T/N
    B = exp(-r*delta)
    q = (B**(-1) - d)/(u-d)
    
    # Precios al tiempo N
    S = S0*(u**(arange(N,-1,-1)))*(d**(arange(0,N+1,1)))
    
    # Valor del payoff
    C = maximum((S-K)* ((-1)**put),0)   
    
    if hedge:
        delta = []
        # Valor del derivado tomando el árbol hacia atrás
        for i in arange(N-1, -1,-1):
            delta.append((C[0:i+1]-C[1:i+2])/(S[0:i+1]-S[1:i+2]))
            S = S0*(u**(arange(i,-1,-1)))*(d**(arange(0,i+1,1)))
            C[:i+1] = B*((1-q)*C[1:i+2] + q*C[0:i+1])
            C = C[:-1]
            C = maximum((S-K)* ((-1)**put),C)
        return C[0], delta
    else:
        # Valor del derivado tomando el árbol hacia atrás
        for i in arange(N-1, -1,-1):
            S = S0*(u**(arange(i,-1,-1)))*(d**(arange(0,i+1,1)))
            C[:i+1] = B*((1-q)*C[1:i+2] + q*C[0:i+1])
            C = C[:-1]
            C = maximum((S-K)* ((-1)**put),C)
        return C[0]
    

def AsianOption(K, T, S0, N, r, u, d):
    # Declaración de variables
    delta = T/N
    B = exp(-r*delta)
    q = (B**(-1) - d)/(u-d)
    S = [S0*(d**(arange(0, i+1, 1)))*(u**(arange(i,-1,-1))) for i in arange(N, -1,-1)]
    up_down=list(itertools.product(['u','d'],repeat=N))
    up_down = np.matrix(up_down)
    n,m = up_down.shape
    M = [[S0]*n]
    for i in range(m):
        temp = up_down[:,:i+1]
        l = []
        for j in range(n):
            x,y = np.unique(np.asarray(temp[j]), return_counts=True)
            if len(x) == 2:
                if x[0] == "u":
                    nu = y[0]
                    nd = y[1]
                else:
                    nd = y[0]
                    nu = y[1]
            else:
                if x[0] == "u":
                    nu = y[0]
                    nd = 0
                else:
                    nd = y[0]
                    nu = 0
            l.append(S0*(u**nu) * (d**nd))
        M.append(l)
    M = np.matrix(M)
    Y = M.sum(axis = 0)
    C = maximum((Y/(N+1))-K, 0)
    C = np.asarray(C)
    C = C[0] # Payoffs

    deltas = []
    for i in arange(N,0,-1):
        m = len(S[N+1-i])
        l = []
        X = []
        chunks = list(partition(2**(N-m),np.asarray(M[m,:]).tolist()[0]))
        for j in range(2**(m-1)):
            temp = chunks[2*j][0] - chunks[2*j+1][0]
            x = (C[2*j]-C[2*j +1])  / (temp)
            l.append(x)
            X.append(B*(q*C[2*j] + (1-q)*C[2*j +1]))
        deltas.append(l)
        C = X
    return C, deltas

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



def LookBackOption(T, S0, N, r, u, d):
    # Declaración de variables
    delta = T/N
    #delta = 1/12
    B = exp(-r*delta)
    q = (B**(-1) - d)/(u-d)
    S = [S0*(d**(arange(0, i+1, 1)))*(u**(arange(i,-1,-1))) for i in arange(N, -1,-1)]
    up_down=list(itertools.product(['u','d'],repeat=N))
    up_down = np.matrix(up_down)
    n,m = up_down.shape
    M = [[S0]*n]
    for i in range(m):
        temp = up_down[:,:i+1]
        l = []
        for j in range(n):
            x,y = np.unique(np.asarray(temp[j]), return_counts=True)
            if len(x) == 2:
                if x[0] == "u":
                    nu = y[0]
                    nd = y[1]
                else:
                    nd = y[0]
                    nu = y[1]
            else:
                if x[0] == "u":
                    nu = y[0]
                    nd = 0
                else:
                    nd = y[0]
                    nu = 0
            l.append(S0*(u**nu) * (d**nd))
        M.append(l)
    M = np.matrix(M)
    C = M.max(0)
    C = np.asarray(C)
    C = C[0] # Payoffs

    deltas = []
    for i in arange(N,0,-1):
        m = len(S[N+1-i])
        l = []
        X = []
        chunks = list(partition(2**(N-m),np.asarray(M[m,:]).tolist()[0]))
        for j in range(2**(m-1)):
            temp = chunks[2*j][0] - chunks[2*j+1][0]
            x = (C[2*j]-C[2*j +1])  / (temp)
            l.append(x)
            X.append(B*(q*C[2*j] + (1-q)*C[2*j +1]))
        deltas.append(l)
        C = X
    return C, deltas


def DigitalOption(K, T, r, S0, N, u, d):
    delta_t = T / N
    # Probabilidad de aumento en el precio
    p = (math.exp(r * delta_t) - d) / (u - d) 
    
    # Construcción árbol del subyacente.
    # Primer paso
    arbol = np.zeros((N + 1, N + 1))
    arbol[0,0] = S0
    # Resto de los pasos
    for col in range(1, N +1):
        for ren in range(0, N +1):
            if((col - ren) >= 0):
                arbol[ren, col] = S0 *(( u ** (col - ren)) * (d ** (ren)))

    # Payoff primer paso
    payoffs = np.zeros((N+1, N+1))
    for ren in range(0, N+1):
        if (arbol[ren, N] > S0):
            payoffs[ren, N] = K
        else:
            payoffs[ren, N] = 0
            
    # Valuando backwards        
    for i in range(1, N + 1):
        col = N - i
    for j in range(0, col +1): 
        payoffs[j, col] = math.exp(-r * delta_t) * (p * payoffs[j, col + 1] + (1 - p)*payoffs[j +1, col +1])
    return payoffs[0,0]    

    # Alfas
    alfas = np.zeros((N, N ))
    for ren in range(0,N):
        for col in range(0, N):
            if((col - ren) >= 0):
                alfas[ren,col]=(payoffs[ren,col+1]-payoffs[ren+1,col+1])/(arbol[ren,col+1]-arbol[ren+1,col+1])


    # Betas
    betas = np.zeros((N, N ))
    for ren in range(0,N):
        for col in range(0, N):
        #Condicional para limitar matriz superior triangular
            if((col - ren) >= 0):
                betas[ren,col]=math.exp(-r * delta_t)*(payoffs[ren,col+1]-((payoffs[ren,col+1]-payoffs[ren+1,col+1])*arbol[ren,col+1])/(arbol[ren,col+1]-arbol[ren+1,col+1]))
    return payoffs[0,0], alfas, betas

class Derivative:
    
    def __init__(self, S0, K, vol, r, N, T, kind, put = False, B = None):
        self.strike = K
        self.kind = kind
        self.put = put
        self.volatility = vol
        self.rate = r
        self.periods = N
        self.lenght = T
        self.deltas = []
        self.S0 = S0
        self.q = 0.0
        self.price = 0.0
        self.barrier = B
        
    def compute(self):
        delta = self.lenght/self.periods
        u = exp((self.rate - ((self.volatility**2)/2))*delta + self.volatility*np.sqrt(delta))
        d = exp((self.rate - ((self.volatility**2)/2))*delta - self.volatility*np.sqrt(delta))
        if self.kind == "European":
            self.price, self.deltas = VanillaEuropeanOption(K = self.strike,\
                                                            T = self.lenght,\
                                                            S0 = self.S0,\
                                                            r = self.rate,\
                                                            N = self.periods,\
                                                            u=u, d=d,\
                                                            put = self.put,\
                                                            hedge = True)
        elif self.kind == "American":
            self.price, self.deltas = VanillaAmericanOption(K = self.strike,\
                                                            T = self.lenght,\
                                                            S0 = self.S0,\
                                                            r = self.rate,\
                                                            N = self.periods,\
                                                            u=u, d=d,\
                                                            put = self.put,\
                                                            hedge = True)
        elif self.kind == "Asian":
            self.price, self.deltas = AsianOption(K = self.strike,\
                                                     T = self.lenght,\
                                                     S0 = self.S0,\
                                                     N = self.periods,\
                                                     r = self.rate,\
                                                     u=u, d=d)
        elif self.kind == "Forward":
            self.price = Forward(K=self.strike, S0=self.S0, r=self.rate)
        elif self.kind == "UpNOut":
            self.price = UpNOutOption(K=self.strike, T=self.lenght, S0=self.S0,\
                                      B=self.barrier, r=self.rate, N=self.periods,\
                                      u = u, d=d, put = self.put)
        elif self.kind == "UpNIN":
            self.price = UpNInOption(K=self.strike, T=self.lenght, S0=self.S0,\
                                     B=self.barrier, r=self.rate, N=self.periods,\
                                     u=u, d=d, put = self.put)
        elif self.kind == "Digital":
            self.price = DigitalOption(K=self.strike, T=self.lenght, r=self.rate,\ 
                                       S0=self.S0, N=self.periods, u=u, d=d):
        elif self.kind == "LookBack":
            self.price, self.deltas = LookBackOption(T=self.lenght, S0=self.S0,\
                                                     N = self.periods, r=self.rate,\
                                                    u=u, d=d)
        else:
            self.price, self.deltas, _ = LookBackOption(T=self.lenght, S0=self.S0,\
                                                        N=self.periods, r=self.rate, u=u, d=d)
        return
    
    def summary(self):
        print("RESUMEN:\
              \n+++++++++++++++++++++++++++\
              \nTipo de Derivado: {0}\
              \nEs un Put?: {1}\
              \nStrike: {2}\
              \nS0: {3}\
              \nVolatilidad:{4}\
              \nTasa: {5}\
              \nLonguitud del periodo de Valuación: {6}\
              \nCantidad de periodos: {7}\
              \nBarrera: {8}\
              \nPrecio: {9}\
              \nDeltas: {10}".format(self.kind, self.put, self.strike, self.S0,\
                  self.volatility,self.rate, self.lenght, self.periods, self.barrier,\
                      self.price, self.deltas))
        return
    
