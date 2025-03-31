"""
Ingenieria Biomedica
Modelado de Sistemas Fisiologicos
Mario Tolentino Cipriano Camacho - 20211961
Dr. Paul Antonio Valle Trujillo
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats
import statistics as st
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv")

T = data.iloc[:,0]; to = T.to_numpy()
x1 = data.iloc[:,1];  x1o= x1.to_numpy()
x2 = data.iloc[:,2]; x2o = x2.to_numpy()
xo = np.mean([x1o,x2o], axis = 0)

print(data)

def plotdata(t,x1,x2,xo):
    fig = plt.figure(); #plt.rcParams['text.use'] = True
    plt.rcParams.update({'font.size':11})
    fig = plt.figure(figsize=(8,4))
    plt.plot(t,x1,linestyle = 'none', marker = 'x', color = [0.11,0.30,0.42], label = "$x_1(t)")
    plt.plot(t,x2,linestyle = 'none', marker = 'x', color = [0.65,0.25,0.25], label = "$x_2(t)")
    plt.plot(t,xo,linewidth = 0.5, marker = 'x', color = [0.25,0.25,0.25], label = "$x_o(t)")
    plt.xlabel('$t$ $[months]$')
    plt.ylabel('$x_i$ $[cells]$')
    plt.xlim(0,72)
    xticks = np.arange(0,75,8)
    plt.xticks(xticks)
    plt.ylim(0,11E8)
    yticks = np.arange(0,12E8,1E8)
    plt.yticks(yticks)
    plt.legend(bbox_to_anchor = (1.01, 1.03), fontsize = 10, title = "$Experimental$ $data$", frameon = True)

    plt.show()
    fig.tight_layout()
    fig.savefig('python_data.pdf')

plotdata(to,x1o,x2o,xo)

def mdl(to,xo,k0,b,S):
    def sigmoidal(t,k):
        dt = 1E-1
        n = round(max(t)/dt)
        time = np.linspace(0,max(t),n+1)
        x = np.zeros(n+1); x[0] = xo[0]

        def f(x):
            if S == 1:
                dx = k*x*(1 - b*x)
            elif S == 2:
                dx = k*x**(2/3)*(1 - b*x**(1/3))
            elif S == 3:
                dx = k*x**(3/4)*(1 - b*x**(1/4))
            elif S == 4:
                dx = k*x*(1 - b*np.log(x))
            elif S == 5:
                dx = k*x*np.log(b/x)
            return dx
        
        for i in range(0,n):
            fx = f(x[i])
            xn = x[i] + fx*dt
            fxn = f(xn)
            x[i+1] = x[i] + (fx +fxn)*dt/2

        xi = np.zeros(len(t))

        for i in range(0,len(t)):
            k = abs(time-t[i]) < 1E-9
            xi[i] = x[k]
        return xi
    npar = len(k0)
    low = np.ones(npar)*math.inf*(-1)
    sup = np.ones(npar)*math.inf

    Estimate,cov = curve_fit(sigmoidal,to,xo,k0,bounds=(low,sup))
    k = Estimate[0]
    xi = sigmoidal(to,k)

    return xi,Estimate,cov

k0 = [0.001]
xmax = np.max(xo)
b = [1/xmax,
     xmax**(-1/3),
     xmax**(-1/4),
     1/np.log(xmax),
     xmax]

def biostatistics(Estimate,cov,xo,xa):
    alpha = 0.05
    dof = len(xo)-len(Estimate)
    tval = scipy.stats.t.ppf(q = 1-alpha/2, df = dof)
    SE = np.diag(cov)**(0.5)
    pvalue = 2*scipy.stats.t.sf(np.abs(Estimate/SE),dof)
    MoE = SE*tval
    CI95 = np.zeros([len(Estimate),2])
    for i in range (0,len(Estimate)):
        CI95[i,0] = Estimate[i]-MoE[i]
        CI95[i,1] = Estimate[i]+MoE[i]
    print('\nStatistics results:\n')
    Parameter = ['k']
    df = pd.DataFrame(list(zip(Parameter,Estimate,SE,MoE,CI95,pvalue)), columns=['Parameter','Estimate','SE','MoE','CI95','pValue'])
    print(df.to_string(index = False))

def plotdatai(t,xo,xa):
    fig = plt.figure(); #plt.rcParams['text.use'] = True
    plt.rcParams.update({'font.size':11})
    fig = plt.figure(figsize=(8,4))
    plt.plot(t,xo,linestyle = 'none', marker = 'x', color = [0.11,0.30,0.42], label = "$x_1(t)")
    plt.plot(t,xa,linewidth = 0.5, marker = 'x', color = [0.25,0.25,0.25], label = "$x_o(t)")
    plt.xlabel('$t$ $[months]$')
    plt.ylabel('$x_i$ $[cells]$')
    plt.xlim(0,72)
    xticks = np.arange(0,75,8)
    plt.xticks(xticks)
    plt.ylim(0,11E8)
    yticks = np.arange(0,12E8,1E8)
    plt.yticks(yticks)
    plt.legend(bbox_to_anchor = (1.01, 1.03), fontsize = 10, title = "$Experimental$ $data$", frameon = True)

    plt.show()
    fig.tight_layout()
    if S == 1:
        fig.savefig('Logistic.pdf')
    elif S == 2:
        fig.savefig('Allometric Spheare.pdf')
    elif S == 3:
        fig.savefig('Alometric Fractal.pdf')
    elif S == 4:
        fig.savefig('Gomperts Normal.pdf')
    elif S == 5:
        fig.savefig('Gomperts Simplified.pdf')


for i in range(0,len(b)):
    S = i+1
    xa,Estimate,cov = mdl(to,xo,k0,b[i],S)
    #print('/nk = ', Estimate)
    print(S)
    plotdatai(to,xo,xa)
    biostatistics(Estimate,cov,xo,xa)
