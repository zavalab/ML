import numpy as np
import matplotlib.pyplot as plt

from munch import Munch
from pyomo.environ import (ConcreteModel, Constraint, Objective, Var, Param, TransformationFactory, value)
from pyomo.dae import ContinuousSet, DerivativeVar

def timeParams(self,t):
    BE = np.zeros(2)
    BE[0] = self.EnergyProgram(t)
    BE[1] = (self.gamma*BE[0] + (1-self.gamma)*self.delta)[0]
    
    # Enthalpy
    delH = np.zeros(3)

    delH[0] = -BE[0]
    delH[1] = BE[0] - BE[1]
    delH[2] = BE[1]
    
    #print(f'delH{delH}')
    
    # Entropy
    delS = np.zeros(3)
    delS[0] = -0.0014
    delS[2] = 0.0014

    delG = delH - self.T*delS

    Ea = self.alpha*delH[1] + self.beta

    k = np.zeros([3,2])
    K = np.zeros(3)

    k[0,0] = (self.kB*self.T/self.h)*np.exp(delS[0]/self.kB)
    k[1,0] = (self.kB*self.T/self.h)*np.exp(-Ea[0]/(self.kB*self.T))
    k[2,1] = (self.kB*self.T/self.h)*np.exp(-delS[2]/self.kB)
    
    K = np.exp(-(delH-self.T*delS)/(self.kB*self.T))
    #print(f'K{K}')

    k[0,1] = k[0,0] / K[0]
    k[1,1] = k[1,0] / K[1]
    k[2,0] = k[2,1] * K[2]

    kf = k[:,0]
    kr = k[:,1]
    
    # if t==0:
    #     print(f'BEa:{BE[0]}')
    #     print(f'BEa:{BE[1]}')
    #     print(f'delH:{delH}')
    #     print(f'delS:{delS}')
    #     print(f'delG:{delG}')
    #     print(f'Ea{Ea}')
    #     print(f'K:{K}')
    #     print(f'kf:{kf}')
    #     print(f'kr:{kr}')

    return Munch(kf=kf,kr=kr,BEa=BE[0])

def calc_rates(self,m,pt,t):

    CAg = m.CAg[t]
    CBg = m.CBg[t]
    CA = m.CA[t]
    CB = m.CB[t]
    Theta = 1 - CA - CB

    kf = pt.kf
    kr = pt.kr

    r = [(kf[0]*(CAg*self.Rg*self.T/self.P0)*Theta - kr[0]*CA),
        (kf[1]*CA - kr[1]*CB),
        (kf[2]*CB - kr[2]*(CBg*self.Rg*self.T/self.P0)*Theta),
        ]
    
    return r

def create_pyomo_model(self):
    m = ConcreteModel()

    # Time Variables
    m.t = ContinuousSet(bounds=(0,1*self.nperiods))
    m.tf = Param(initialize = self.nperiods * 1 / self.fmin)

    # Flowrate variables
    m.q = Param(initialize = self.qdot, mutable=True)

    # Intial Conc Variables
    m.CA0 = Param(initialize = 1)
    m.CB0 = Param(initialize = 0)

    # Gas-phase concentrations
    m.CAg = Var(m.t, bounds = (0,1))
    m.CBg = Var(m.t, bounds = (0,1))

    # Surface-phase concentrations
    m.CA = Var(m.t, bounds = (0,1))
    m.CB = Var(m.t, bounds = (0,1))

    # Gas-phase concentration derivatives
    m.dCAgdt = DerivativeVar(m.CAg)
    m.dCBgdt = DerivativeVar(m.CBg)

    # Surface-phase concentration derivatives
    m.dCAdt = DerivativeVar(m.CA)
    m.dCBdt = DerivativeVar(m.CB)

    @m.Constraint(m.t)
    def dAgdt_eq(m,t):
        if t == m.t.first():
            return Constraint.Skip
        pt = self.timeParams(t)
        r = self.calc_rates(m,pt,t)
        return m.dCAgdt[t] == m.tf*(-r[0]*(self.Nsites/self.V) + m.q/self.V*(m.CA0-m.CAg[t]))

    @m.Constraint(m.t)
    def dBgdt_eq(m,t):
        if t == m.t.first():
            return Constraint.Skip
        pt = self.timeParams(t)
        r = self.calc_rates(m,pt,t)
        return m.dCBgdt[t] == m.tf*(r[2]*(self.Nsites/self.V) + m.q/self.V*(m.CB0-m.CBg[t]))

    @m.Constraint(m.t)
    def dAdt_eq(m,t):
        if t == m.t.first():
            return Constraint.Skip
        pt = self.timeParams(t)
        r = self.calc_rates(m,pt,t)
        return m.dCAdt[t] == m.tf*(r[0] - r[1])

    @m.Constraint(m.t)
    def dBdt_eq(m,t):
        if t == m.t.first():
            return Constraint.Skip
        pt = self.timeParams(t)
        r = self.calc_rates(m,pt,t)
        return m.dCBdt[t] == m.tf*(r[1] - r[2])
    
    m.CAgCSS = Constraint(expr = m.CAg[m.t.first()] == m.CAg[m.t.last()])
    m.CBgCSS = Constraint(expr = m.CBg[m.t.first()] == m.CBg[m.t.last()])

    m.CACSS = Constraint(expr = m.CA[m.t.first()] == m.CA[m.t.last()])
    m.CBCSS = Constraint(expr = m.CB[m.t.first()] == m.CB[m.t.last()])
    
    m.obj = Objective(expr = (m.CAg[m.t.first()] - m.CAg[m.t.last()])**2 + 
                      (m.CBg[m.t.first()] - m.CBg[m.t.last()])**2 +
                      (m.CA[m.t.first()] - m.CA[m.t.last()])**2 +
                      (m.CB[m.t.first()] - m.CB[m.t.last()])**2)
    
    # Return the model
    return m

def InitializeModel(self, tspan, warmstart=False):
    set_times = list(self.m.t)
    
    if warmstart == False:
        #print('Fresh Start')
        for i in range(self.nfe):
            self.m.CAg[set_times[i]].value = 0.99
            self.m.CBg[set_times[i]].value = 0.01
    
            self.m.CA[set_times[i]].value = 1
            self.m.CB[set_times[i]].value = 0
    
            self.m.dCAgdt[set_times[i]].value = 0
            self.m.dCBgdt[set_times[i]].value = 0
    
            self.m.dCAdt[set_times[i]].value = 0
            self.m.dCBdt[set_times[i]].value = 0

    else:
        #print('Warm Start')
        for i in range(self.nfe):
            self.m.CAg[set_times[i]].value = np.clip(self.CAg[i],0,1)
            self.m.CBg[set_times[i]].value = np.clip(self.CBg[i],0,1)
            
            self.m.CA[set_times[i]].value = np.clip(self.CA[i],0,1)
            self.m.CB[set_times[i]].value = np.clip(self.CB[i],0,1)
            
            self.m.dCAgdt[set_times[i]].value = self.dCAg[i]
            self.m.dCBgdt[set_times[i]].value = self.dCBg[i]
    
            self.m.dCAdt[set_times[i]].value = self.dCA[i]
            self.m.dCBdt[set_times[i]].value = self.dCB[i]

def getConcData(self,m):
    tdata = np.array(sorted(m.t))
    self.tdata = tdata
    
    self.CAg = np.clip(np.array([value(m.CAg[t]) for t in m.t]),1e-12,1)
    self.CBg = np.clip(np.array([value(m.CBg[t]) for t in m.t]),1e-12,1)
    
    self.CA = np.clip(np.array([value(m.CA[t]) for t in m.t]),1e-12,1)
    self.CB = np.clip(np.array([value(m.CB[t]) for t in m.t]),1e-12,1)
    
    self.dCAg = np.array([value(m.dCAgdt[t]) for t in m.t])
    self.dCBg = np.array([value(m.dCBgdt[t]) for t in m.t])
    
    self.dCA = np.array([value(m.dCAdt[t]) for t in m.t])
    self.dCB = np.array([value(m.dCBdt[t]) for t in m.t])
    
    self.m = m
    
def InterpConc(self,tspan):
    self.CAg = np.interp(tspan, self.tdata, self.CAg)
    self.CBg = np.interp(tspan, self.tdata, self.CBg)
    self.CA = np.interp(tspan, self.tdata, self.CA)
    self.CB = np.interp(tspan, self.tdata, self.CB)
    self.dCAg = np.interp(tspan, self.tdata, self.dCAg)
    self.dCBg = np.interp(tspan, self.tdata, self.dCBg)
    self.dCA = np.interp(tspan, self.tdata, self.dCA)
    self.dCB = np.interp(tspan, self.tdata, self.dCB)
    
def printSimResults(self):
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, figsize=(16,9))
    
    tplot = self.tdata*1/self.fmin
    units = 's'
    if self.fmin >= 1e2:
        tplot = tplot*1000
        units = 'ms'
    # Concentration A gas
    ax1.plot(tplot,self.CAg)
    ax1.set_title('CAg')
    ax1.set_xlabel(f'Time ({units})')
    ax1.set_ylabel('Concentration (mol/L)')

    ax2.plot(tplot,self.CBg)
    ax2.set_title('Cprod')
    ax2.set_xlabel(f'Time ({units})')
    ax2.set_ylabel('Concentration (mol/L)')

    ax3.plot(tplot,self.CA)
    ax3.plot(tplot,self.CB)
    ax3.set_title('Csurf')
    ax3.set_xlabel(f'Time ({units})')
    ax3.set_ylabel('Surface Fraction')
    ax3.legend(['A','B'])

    ax4.plot(tplot,self.dCAg)
    ax4.set_title('dCAg')
    ax4.set_xlabel(f'Time ({units})')
    ax4.set_ylabel('dConc/dt (mol/L*t)')
    
    ax5.plot(tplot,self.dCBg)
    ax5.set_title('dCprod')
    ax5.set_xlabel(f'Time ({units})')
    ax5.set_ylabel('dConc/dt (mol/L*t)')
    
    ax6.plot(tplot,self.dCA)
    ax6.plot(tplot,self.dCB)
    ax6.set_title('dCsurf')
    ax6.set_xlabel(f'Time ({units})')
    ax6.set_ylabel('dConc/dt (mol/L*t)')
    ax6.legend(['A','B'])
    
    fig.tight_layout()