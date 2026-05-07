from munch import Munch
import numpy as np
from pyomo.environ import (ConcreteModel, Constraint, Objective, Var, Param,
                           SolverFactory, value, sin, cos, Set, Suffix,
                           TransformationFactory, assert_optimal_termination, RangeSet, NonNegativeReals)
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm 
from scipy.optimize import minimize
import time
import torch
import matplotlib as mpl
import networkx as nx
import matplotlib.animation as animation
import os
import imageio.v3 as iio
import tempfile
import random
import itertools
import gpytorch
import botorch
from os.path import exists
import pathlib
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf

class ProCatModel():
    # Contain all the information needed to simulate a DCN model to CSS
    
    def __init__(self, NewParams=False):
        self.T = 150+273.15
        self.Ca_target = 0.01
        self.qdot = 1
        self.gamma = np.array([1.16,1.00,1.05,1.37,1.23])
        self.delta = np.array([0.79,0.82,0.55,0.58,0.90])
        self.alpha = np.array([1.12,0.87,0.97,0.81,1.13,0.67,0.63,0.68,1.16])
        self.beta = np.array([0.43,0.57,0.61,0.54,0.21,0.77,0.64,0.56,0.46])
        self.BEa = 0.6 # [eV]
        self.Nsites = 2.76e-6
        self.V = 2.60e-4
        self.P0 = 1.01325
        self.R = 8.314
        self.nfe = 300
        self.number_periods = 1
        self.kB = 8.61733034e-5
        self.h = 4.1357e-15
        self.Rg = 0.08314
        
        if NewParams == True:
            self.gamma = np.loadtxt('gamma.csv')
            self.delta = np.loadtxt('delta.csv')
            self.alpha = np.loadtxt('alpha.csv')
            self.beta = np.loadtxt('beta.csv')
            self.BEa = np.loadtxt('BEa.csv')
            
    def EnergyProgram(self,t):
        if np.size(t)==1:
            nt = 1
        else:
            nt = len(t)
        E = np.zeros([self.nsigs,nt])
        
        if self.SigShape == 'Square':
            for i in range(self.nsigs):
                E[i,:] = -self.Amp[i]/2*signal.square(2*np.pi*1*t,duty=self.DC[i])
            Esigs = np.concatenate([E,self.Offset*np.ones([1,nt])],axis=0)
        elif self.SigShape == 'Sine':
            for i in range(self.nsigs):
                E[i,:] = self.Amp[i]*(np.sin(2*np.pi*(i+1)*t))
            Esigs = np.concatenate([E,self.Offset*np.ones([1,nt])],axis=0)
        elif self.SigShape == 'Logistic':
            for i in range(self.nsigs):
               LK = 250
               LO = -self.Amp[i]/2+self.Offset
               
               if self.DC[i] == 0:
                   E[i,:] = np.ones([1,nt])*self.Amp[i]/2+self.Offset
               elif self.DC[i] == 1:
                   E[i,:] = np.ones([1,nt])*-self.Amp[i]/2+self.Offset
               else:
                   E[i,:] = self.Amp[i]/(1+np.exp(-LK*(t-self.DC[i]))) - 2*self.Amp[i]/(1+np.exp(-LK*(t-1)))+LO
                   E[i,0] = -self.Amp[i]/2+self.Offset
                   E[i,-1] = -self.Amp[i]/2+self.Offset
        
        self.E = E
        Esigs = np.concatenate([E,self.Offset*np.ones([1,nt])],axis=0)
        BEaplot = np.clip(np.sum(E,axis=0) + self.Offset + self.BEa,self.Elb,self.Eub)  
        
        if np.size(t)==1:
            return BEaplot
        else:
            return Esigs, BEaplot
        
        
    def printEnergyProgram(self):
        tspan = np.linspace(0,1,1001)
        Esigs, BEaplot = self.EnergyProgram(tspan)
        tplot = np.linspace(0,1,1001)*1/self.Freq
        
        if np.shape(Esigs)[0] > 2:
            fig, ax = plt.subplots()
            for i in range(self.nsigs):
                ax.plot(tplot,Esigs[i,:],label=f'Sig{i+1}')
            ax.set_title('Component Signals')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Energy (eV)')
            ax.grid()
            ax.legend()
        
        fig, ax = plt.subplots()
        ax.plot(tplot,BEaplot)
        ax.set_title('BEa')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy (eV)')
        ax.grid()
    
    def SetEnergyParams(self,SigParams,SigShape,printSignals=False):
        print(SigParams)
        print(SigShape)
        print(printSignals)
        self.Amp = SigParams['Amplitude']
        self.DC = SigParams['DutyCycle']
        self.Freq = SigParams['Frequency']
        self.Offset = SigParams['Offset']
        self.Elb = SigParams['Ebounds'][0]
        self.Eub = SigParams['Ebounds'][1]
        self.SigShape = SigShape
        self.nsigs = len(self.Amp)
        
        if printSignals == True:
            self.printEnergyProgram()
            
    def timeParams(self,t):
        BE = np.zeros(6)
        BE[0] = self.BEa + self.EnergyProgram(t)
        BE[1:] = np.max(np.vstack([np.zeros(5), self.gamma*BE[0] + (1-self.gamma)*self.delta]), axis = 0)
        
        # Enthalpy
        delH = np.zeros(13)
    
        # Gas species
        delH[0] = -BE[0]
        delH[-3:] = BE[-3:]
    
        # Loop 1
        delH[1] = BE[1] - BE[0]
        delH[2] = BE[2] - BE[1]
        delH[3] = BE[0] - BE[2]
    
        # Loop 2
        delH[4] = BE[3] - BE[1]
        delH[5] = BE[4] - BE[3]
        delH[6] = BE[1] - BE[4]
    
        # Loop 3
        delH[7] = BE[4] - BE[2]
        delH[8] = BE[5] - BE[4]
        delH[9] = BE[2] - BE[5]
        
        # Entropy for surface reactions
        delS = np.zeros(13)
        delS[0] = -0.0014
        delS[10:] = 0.0014
    
        delG = delH - self.T*delS
    
        Ea = np.max(np.vstack([delH[1:10], self.alpha*delH[1:10] + self.beta, np.zeros(9)]) , axis = 0)
    
        #print(f'Ea{Ea}')
    
        k = np.zeros([13,2])
        K = np.zeros(13)
    
        k[0,0] = (self.kB*self.T/self.h)*np.exp(delS[0]/self.kB)
        k[10:,1] = (self.kB*self.T/self.h)*np.exp(-delS[10:]/self.kB)
        k[1:10,0] = (self.kB*self.T/self.h)*np.exp(-Ea/(self.kB*self.T))
    
        K = np.exp(-(delH-self.T*delS)/(self.kB*self.T))
        #print(f'K{K}')
    
        k[0:10,1] = k[0:10,0] / K[0:10]
        k[10:,0] = k[10:,1] * K[10:]
    
        kf = k[:,0]
        kr = k[:,1]
    
        return Munch(kf=kf,kr=kr,BEa=BE[0])
    
    def calc_rates(self,m,pt,t):
    
        CAg = m.CAg[t]
        CDg = m.CDg[t]
        CEg = m.CEg[t]
        CFg = m.CFg[t]
    
        CA = m.CA[t]
        CB = m.CB[t]
        CC = m.CC[t]
        CD = m.CD[t]
        CE = m.CE[t]
        CF = m.CF[t]
        Theta = 1 - CA - CB - CC - CD - CE - CF
    
        kf = pt.kf
        kr = pt.kr
    
        r = [(kf[0]*(CAg*self.Rg*self.T/self.P0)*Theta - kr[0]*CA),
            (kf[1]*CA - kr[1]*CB),
            (kf[2]*CB - kr[2]*CC),
            (kf[3]*CC - kr[3]*CA),
            (kf[4]*CB - kr[4]*CD),
            (kf[5]*CD - kr[5]*CE),
            (kf[6]*CE - kr[6]*CB),
            (kf[7]*CC - kr[7]*CE),
            (kf[8]*CE - kr[8]*CF),
            (kf[9]*CF - kr[9]*CC),
            (kf[10]*CD - kr[10]*(CDg*self.Rg*self.T/self.P0)*Theta),
            (kf[11]*CE - kr[11]*(CEg*self.Rg*self.T/self.P0)*Theta),
            (kf[12]*CF - kr[12]*(CFg*self.Rg*self.T/self.P0)*Theta),
            ]
        
        return r
    
    def create_pyomo_model(self, qinit=None, Cinit = None):
    
        # Handle qin
        if qinit==None:
            q0 = self.qdot
        else:
            q0 = qinit
        
        #print(q0)
        
        # Handle Cin
        if Cinit == None:
            CAg0 = 0.99*np.ones(self.nfe+1)
            CDg0 = 0.0033*np.ones(self.nfe+1)
            CEg0 = 0.0033*np.ones(self.nfe+1)
            CFg0 = 0.0033*np.ones(self.nfe+1)
            CA0 = 0.15*np.ones(self.nfe+1)
            CB0 = 0.15*np.ones(self.nfe+1)
            CC0 = 0.15*np.ones(self.nfe+1)
            CD0 = 0.15*np.ones(self.nfe+1)
            CE0 = 0.15*np.ones(self.nfe+1)
            CF0 = 0.15*np.ones(self.nfe+1)
            dCAg0 = 0*np.ones(self.nfe+1)
            dCDg0 = 0*np.ones(self.nfe+1)
            dCEg0 = 0*np.ones(self.nfe+1)
            dCFg0 = 0*np.ones(self.nfe+1)
            dCA0 = 0*np.ones(self.nfe+1)
            dCB0 = 0*np.ones(self.nfe+1)
            dCC0 = 0*np.ones(self.nfe+1)
            dCD0 = 0*np.ones(self.nfe+1)
            dCE0 = 0*np.ones(self.nfe+1)
            dCF0 = 0*np.ones(self.nfe+1)
        else:
            [Cgas,Csurf,Cder] = Cinit
            CAg0 = Cgas[0,:]
            CDg0 = Cgas[1,:]
            CEg0 = Cgas[2,:]
            CFg0 = Cgas[3,:]
            CA0 = Csurf[0,:]
            CB0 = Csurf[1,:]
            CC0 = Csurf[2,:]
            CD0 = Csurf[3,:]
            CE0 = Csurf[4,:]
            CF0 = Csurf[5,:]
            dCAg0 = Cder[0,:]
            dCDg0 = Cder[1,:]
            dCEg0 = Cder[2,:]
            dCFg0 = Cder[3,:]
            dCA0 = Cder[4,:]
            dCB0 = Cder[5,:]
            dCC0 = Cder[6,:]
            dCD0 = Cder[7,:]
            dCE0 = Cder[8,:]
            dCF0 = Cder[9,:]
        
    
        m = ConcreteModel()
    
        # Time Variables
        m.tau = ContinuousSet(bounds=(0,1))
        m.times = Var(m.tau)
        m.tf = Param(initialize = 1/self.Freq)
    
        # Flowrate variables
        m.q = Param(initialize = q0, mutable=True)
    
        # Intial Conc Variables
        m.CA0 = Param(initialize = 1)
        m.CD0 = Param(initialize = 0)
        m.CE0 = Param(initialize = 0)
        m.CF0 = Param(initialize = 0)
    
        # Gas-phase concentrations
        m.CAg = Var(m.tau, bounds = (0,1))
        m.CDg = Var(m.tau, bounds = (0,1))
        m.CEg = Var(m.tau, bounds = (0,1))
        m.CFg = Var(m.tau, bounds = (0,1))
    
        # Surface-phase concentrations
        m.CA = Var(m.tau, bounds = (0,1))
        m.CB = Var(m.tau, bounds = (0,1))
        m.CC = Var(m.tau, bounds = (0,1))
        m.CD = Var(m.tau, bounds = (0,1))
        m.CE = Var(m.tau, bounds = (0,1))
        m.CF = Var(m.tau, bounds = (0,1))
        
        # Time scaling
        m.dtimedtau = DerivativeVar(m.times)
    
        # Gas-phase concentration derivatives
        m.dCAgdt = DerivativeVar(m.CAg)
        m.dCDgdt = DerivativeVar(m.CDg)
        m.dCEgdt = DerivativeVar(m.CEg)
        m.dCFgdt = DerivativeVar(m.CFg)
    
        # Surface-phase concentration derivatives
        m.dCAdt = DerivativeVar(m.CA)
        m.dCBdt = DerivativeVar(m.CB)
        m.dCCdt = DerivativeVar(m.CC)
        m.dCDdt = DerivativeVar(m.CD)
        m.dCEdt = DerivativeVar(m.CE)
        m.dCFdt = DerivativeVar(m.CF)
    
        gastol = 1e0
        surftol = 1e0
        sstol = 1e0
        # Time scaling constraint
        @m.Constraint(m.tau)
        def _ode3(m,t):
            if t==0:
                return Constraint.Skip
            return m.dtimedtau[t] == m.tf
    
        @m.Constraint(m.tau)
        def dAgdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCAgdt[t]*gastol == m.tf*(-r[0]*(self.Nsites/self.V) + m.q/self.V*(m.CA0-m.CAg[t]))*gastol
    
        @m.Constraint(m.tau)
        def dDgdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCDgdt[t]*gastol == m.tf*(r[10]*(self.Nsites/self.V) + m.q/self.V*(m.CD0-m.CDg[t]))*gastol
    
        @m.Constraint(m.tau)
        def dEgdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCEgdt[t]*gastol == m.tf*(r[11]*(self.Nsites/self.V) + m.q/self.V*(m.CE0-m.CEg[t]))*gastol
    
        @m.Constraint(m.tau)
        def dFgdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCFgdt[t]*gastol == m.tf*(r[12]*(self.Nsites/self.V) + m.q/self.V*(m.CF0-m.CFg[t]))*gastol
    
        @m.Constraint(m.tau)
        def dAdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCAdt[t]*surftol == m.tf*(r[0] - r[1] + r[3])*surftol
    
        @m.Constraint(m.tau)
        def dBdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCBdt[t]*surftol == m.tf*(r[1] - r[2] - r[4] + r[6])*surftol
    
        @m.Constraint(m.tau)
        def dCdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCCdt[t]*surftol == m.tf*(r[2] - r[3] - r[7] + r[9])*surftol
    
        @m.Constraint(m.tau)
        def dDdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCDdt[t]*surftol == m.tf*(r[4] - r[5] - r[10])*surftol
    
        @m.Constraint(m.tau)
        def dEdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCEdt[t]*surftol == m.tf*(r[5] - r[6] + r[7] - r[8] - r[11])*surftol 
    
        @m.Constraint(m.tau)
        def dFdt_eq(m,t):
            pt = self.timeParams(t)
            r = self.calc_rates(m,pt,t)
            return m.dCFdt[t]*surftol == m.tf*(r[8] - r[9] - r[12])*surftol
    
        # Declare discretization method and parameters 
        discretizer = TransformationFactory('dae.finite_difference')
        discretizer.apply_to(m, nfe = self.nfe, scheme='BACKWARD')
    
        #discretizer = TransformationFactory('dae.collocation')
        #discretizer.apply_to(m, nfe=100, ncp=3, scheme='LAGRANGE-RADAU')
        
        m.CAgCSS = Constraint(expr = m.CAg[m.tau.first()]*sstol == m.CAg[m.tau.last()]*sstol)
        m.CDgCSS = Constraint(expr = m.CDg[m.tau.first()]*sstol == m.CDg[m.tau.last()]*sstol)
        m.CEgCSS = Constraint(expr = m.CEg[m.tau.first()]*sstol == m.CEg[m.tau.last()]*sstol)
        m.CFgCSS = Constraint(expr = m.CFg[m.tau.first()]*sstol == m.CFg[m.tau.last()]*sstol)
    
        m.CACSS = Constraint(expr = m.CA[m.tau.first()]*sstol == m.CA[m.tau.last()]*sstol)
        m.CBCSS = Constraint(expr = m.CB[m.tau.first()]*sstol == m.CB[m.tau.last()]*sstol)
        m.CCCSS = Constraint(expr = m.CC[m.tau.first()]*sstol == m.CC[m.tau.last()]*sstol)
        m.CDCSS = Constraint(expr = m.CD[m.tau.first()]*sstol == m.CD[m.tau.last()]*sstol)
        m.CECSS = Constraint(expr = m.CE[m.tau.first()]*sstol == m.CE[m.tau.last()]*sstol)
        m.CFCSS = Constraint(expr = m.CF[m.tau.first()]*sstol == m.CF[m.tau.last()]*sstol)
    
        #m.MBal = Constraint(expr = 1 - (m.CAg[m.tau.first()] + m.CDg[m.tau.first()]+ m.CEg[m.tau.first()]+ m.CFg[m.tau.first()]) <= 1e-4)
    
        m.obj = Objective(expr = (m.CAg[m.tau.first()] - m.CAg[m.tau.last()])**2 + 
                          (m.CDg[m.tau.first()] - m.CDg[m.tau.last()])**2 +
                          (m.CEg[m.tau.first()] - m.CEg[m.tau.last()])**2 +
                          (m.CFg[m.tau.first()] - m.CFg[m.tau.last()])**2 +
                          (m.CA[m.tau.first()] - m.CA[m.tau.last()])**2 +
                          (m.CB[m.tau.first()] - m.CB[m.tau.last()])**2 +
                          (m.CC[m.tau.first()] - m.CC[m.tau.last()])**2 +
                          (m.CD[m.tau.first()] - m.CD[m.tau.last()])**2 +
                          (m.CE[m.tau.first()] - m.CE[m.tau.last()])**2 +
                          (m.CF[m.tau.first()] - m.CF[m.tau.last()])**2 )
    
        #Initialize variables
        set_times = list(m.tau)
    
        for i in range(self.nfe+1):
            m.CAg[set_times[i]].value = CAg0[i]
            m.CDg[set_times[i]].value = CDg0[i]
            m.CEg[set_times[i]].value = CEg0[i]
            m.CFg[set_times[i]].value = CFg0[i]
    
            m.CA[set_times[i]].value = CA0[i]
            m.CB[set_times[i]].value = CB0[i]
            m.CC[set_times[i]].value = CC0[i]
            m.CD[set_times[i]].value = CD0[i]
            m.CE[set_times[i]].value = CE0[i]
            m.CF[set_times[i]].value = CF0[i]
    
            m.dCAgdt[set_times[i]].value = dCAg0[i]
            m.dCDgdt[set_times[i]].value = dCDg0[i]
            m.dCEgdt[set_times[i]].value = dCEg0[i]
            m.dCFgdt[set_times[i]].value = dCFg0[i]
    
            m.dCAdt[set_times[i]].value = dCA0[i]
            m.dCBdt[set_times[i]].value = dCB0[i]
            m.dCCdt[set_times[i]].value = dCC0[i]
            m.dCDdt[set_times[i]].value = dCD0[i]
            m.dCEdt[set_times[i]].value = dCE0[i]
            m.dCFdt[set_times[i]].value = dCF0[i]
        
        # Return the model
        return m
    
    def concentration_from_decision_vars(self, tee=True):

        m = self.create_pyomo_model()
    
        solver = SolverFactory('ipopt', executable = r'C:\msys64\home\jflory-admin\Ipopt\IpoptCompile3\src\Apps\AmplSolver\ipopt.exe')
        #solver.options['linear_solver'] = 'ma57'
        #solver.options['ma57_automatic_scaling'] = 'yes'
        #solver.options['nlp_scaling_method'] = 'gradient-based'
        solver.options['max_cpu_time'] = 60
        #solver.options['print_info_string'] = 'yes'
        #solver.options['tol'] = 1e-4
        #solver.options['acceptable_tol'] = 1e-1
        #solver.options['constr_viol_tol'] = 1e-2
        #status = solver.solve(m, tee=tee)
        converge = False
        counter = 1
        
        while ((converge == False) and (counter <=5)):
    
            status = solver.solve(m, tee=tee)
            counter = counter + 1
            
            CAg = [value(m.CAg[t]) for t in m.tau]
            CDg = [value(m.CDg[t]) for t in m.tau]
            CEg = [value(m.CEg[t]) for t in m.tau]
            CFg = [value(m.CFg[t]) for t in m.tau]
        
            CAg_avg = np.trapz(CAg,m.tau)
            CDg_avg = np.trapz(CDg,m.tau)
            CEg_avg = np.trapz(CEg,m.tau)
            CFg_avg = np.trapz(CFg,m.tau)
        
            COutg = CDg_avg + CEg_avg + CFg_avg
            
            #print(f'CAg_avg: {CAg_avg}')
            #print(f'COutg_avg: {COutg}')
    
            if abs(CAg_avg - (1-self.Ca_target)) >= 1e-4:
                self.qdot = value(m.q)
                #print(q_old)
                q_new1 = self.qdot*COutg/0.01
                #q_new2 = self.qdot*(1-CAg_avg)/(1-0.99)
                    
                #print(q_new1)
                #print(q_new2)
                m.q = q_new1
                print(value(m.q))
            else:
                converge = True
    
        # Extract Concentration profiles for warm-starting square system
        tdata = np.linspace(0,1,self.nfe+1)*1/self.Freq
        self.tdata = tdata
        
        self.CAg = [value(m.CAg[t]) for t in m.tau]
        self.CDg = [value(m.CDg[t]) for t in m.tau]
        self.CEg = [value(m.CEg[t]) for t in m.tau]
        self.CFg = [value(m.CFg[t]) for t in m.tau]
    
        self.CA = [value(m.CA[t]) for t in m.tau]
        self.CB = [value(m.CB[t]) for t in m.tau]
        self.CC = [value(m.CC[t]) for t in m.tau]
        self.CD = [value(m.CD[t]) for t in m.tau]
        self.CE = [value(m.CE[t]) for t in m.tau]
        self.CF = [value(m.CF[t]) for t in m.tau]
    
        self.dCAg = [value(m.dCAgdt[t]) for t in m.tau]
        self.dCDg = [value(m.dCDgdt[t]) for t in m.tau]
        self.dCEg = [value(m.dCEgdt[t]) for t in m.tau]
        self.dCFg = [value(m.dCFgdt[t]) for t in m.tau]
        self.dCA = [value(m.dCAdt[t]) for t in m.tau]
        self.dCB = [value(m.dCBdt[t]) for t in m.tau]
        self.dCC = [value(m.dCCdt[t]) for t in m.tau]
        self.dCD = [value(m.dCDdt[t]) for t in m.tau]
        self.dCE = [value(m.dCEdt[t]) for t in m.tau]
        self.dCF = [value(m.dCFdt[t]) for t in m.tau]
    
        Sel = self.calc_selectivity('All')
        print(f'Sel: {Sel}')
    
    def calc_selectivity(self, Obj='All'):
        CAg_avg = np.mean(self.CAg)
        CDg_avg = np.mean(self.CDg)
        CEg_avg = np.mean(self.CEg)
        CFg_avg = np.mean(self.CFg)
    
        Sel = np.array([CDg_avg/CAg_avg,
                        CEg_avg/CAg_avg,
                        CFg_avg/CAg_avg])
    
        Sel = Sel/np.sum(Sel)*100
    
        if Obj == 'D':
            return Sel[0]
        elif Obj == 'E':
            return Sel[1]
        elif Obj == 'F':
            return Sel[2]
        else:
            return Sel
        
    def printSimResults(self):
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, figsize=(16,9))
    
        # Concentration A gas
        ax1.plot(self.tdata,self.CAg)
        ax1.set_title('CAg')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Concentration (mol/L)')
    
        ax2.plot(self.tdata,self.CDg)
        ax2.plot(self.tdata,self.CEg)
        ax2.plot(self.tdata,self.CFg)
        ax2.set_title('Cprod')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Concentration (mol/L)')
        ax2.legend(['D','E','F'])
    
        ax3.plot(self.tdata,self.CA)
        ax3.plot(self.tdata,self.CB)
        ax3.plot(self.tdata,self.CC)
        ax3.plot(self.tdata,self.CD)
        ax3.plot(self.tdata,self.CE)
        ax3.plot(self.tdata,self.CF)
        ax3.set_title('Csurf')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Surface Fraction')
        ax3.legend(['A','B','C','D','E','F'])
    
        ax4.plot(self.tdata,self.dCAg)
        ax4.set_title('dCAg')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('dConc/dt (mol/L*t)')
        
        ax5.plot(self.tdata,self.dCDg)
        ax5.plot(self.tdata,self.dCEg)
        ax5.plot(self.tdata,self.dCFg)
        ax5.set_title('dCprod')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('dConc/dt (mol/L*t)')
        ax5.legend(['D','E','F'])
    
        ax6.plot(self.tdata,self.dCA)
        ax6.plot(self.tdata,self.dCB)
        ax6.plot(self.tdata,self.dCC)
        ax6.plot(self.tdata,self.dCD)
        ax6.plot(self.tdata,self.dCE)
        ax6.plot(self.tdata,self.dCF)
        ax6.set_title('dCsurf')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('dConc/dt (mol/L*t)')
        ax6.legend(['A','B','C','D','E','F'])
    
        n_cycles = 1
        tplot = np.linspace(0,n_cycles,self.nfe*n_cycles)*1/self.Freq
        Dplot = np.tile(self.CDg[:-1],n_cycles)
        Eplot = np.tile(self.CEg[:-1],n_cycles)
        Fplot = np.tile(self.CFg[:-1],n_cycles)
        fig, ax = plt.subplots(1,1)
        ax.plot(tplot,Dplot)
        ax.plot(tplot,Eplot)
        ax.plot(tplot,Fplot)
        ax.set_title('Product Gas Concentrations')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration (mol/L)')
        ax.legend(['D','E','F'],loc='upper right')
        ax.grid()
        #fig.savefig('ProductConcTestLog.png')

            
if __name__ == '__main__':
    eparams = {'Amplitude':[1,-0.25],
               'DutyCycle':[0.5,0.25],
               'Frequency':1e3,
               'Offset':0.0,
               'Ebounds':[0.5,1.5]
        }
    
    dcn = ProCatModel()
    dcn.SetEnergyParams(eparams,'Logistic',True)
    dcn.create_pyomo_model()
    dcn.concentration_from_decision_vars()
    dcn.printSimResults()