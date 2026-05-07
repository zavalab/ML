import numpy as np
from pyomo.environ import SolverFactory, value, TransformationFactory
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import qmc
import torch
import itertools
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qUpperConfidenceBound, qExpectedImprovement
from botorch.optim import optimize_acqf

# Overall class which contains all necessary information for the system
class ProCatModel():

    # Set up all default parameters and load surface dynamics parameters
    def __init__(self, BEParams, EaParams):
        
        
        # Default reactor parameter values
        self.T = 423.15 # [K]
        self.qdot = 1 # [L/s]
        self.Nsites = 2.76e-6 # [gmol]
        self.V = 2.60e-4 #[L]
        self.P0 = 1.01325 # [bar]
        self.R = 8.314 #[J/molK]
        self.kB = 8.61733034e-5 # [eV/K]
        self.h = 4.1357e-15 # [eV*s]
        self.Rg = 0.08314 # [L-bar/mol-K]

        # Default simulation parameters
        self.nfe = 100 
        self.number_periods = 1
        self.nperiods = 1

        # Process binding energy and activation energy parameters
        self.gamma = BEParams[0,:].reshape(-1)
        self.delta = BEParams[1,:].reshape(-1)
        self.alpha = EaParams[0,:].reshape(-1)
        self.beta = EaParams[1,:].reshape(-1)

        # Load model specific functions from model file
        importSystemFuncs()
        
    def EnergyProgram(self,t):
        """
        Constructs the time series energy program values based on the signal parameters

        Args: 
            t (nparray): time when the energy program should be constructed

        Returns:
            BEaplot (nparray): Binding Energy of A* at specified time(s)
            Esigs (double or nparray): Component basis function signal values (not returned if len(t) is 1)           
        """
        
        if np.size(t)==1:
            nt = 1
        else:
            nt = len(t)
        E = np.zeros([self.nsigs,nt])
        
        if self.SigShape == 'Square':
            for i in range(self.nsigs):
                E[i,:] = -self.Amp[i]/2*signal.square(2*np.pi*1*t*self.Freq[i]/self.fmin,duty=self.DC[i])
            Esigs = np.concatenate([E,self.Offset*np.ones([1,nt])],axis=0)
        elif self.SigShape == 'Sine':
            for i in range(self.nsigs):
                E[i,:] = -self.Amp[i]/2*(np.sin(2*np.pi*(i+1)*t))
            Esigs = np.concatenate([E,self.Offset*np.ones([1,nt])],axis=0)
    
        self.E = E
        Esigs = np.concatenate([E,self.Offset*np.ones([1,nt])],axis=0)
        BEaplot = np.clip(np.sum(E,axis=0) + self.Offset,self.Elb,self.Eub)  
        
        if np.size(t)==1:
            return BEaplot[0]
        else:
            return Esigs, BEaplot
        
        
    def printEnergyProgram(self):
        """ 
        Prints out the current system energy program
        """
        tspan = np.linspace(0,1,1001)
        Esigs, BEaplot = self.EnergyProgram(tspan)
        
        tplot = tspan*1/self.fmin
        units = 's'
        if self.fmin >= 1e2:
            tplot = tplot*1000
            units = 'ms'
        
        if np.shape(Esigs)[0] > 2:
            fig, ax = plt.subplots()
            for i in range(self.nsigs):
                ax.plot(tplot,Esigs[i,:],label=f'Sig{i+1}')
            ax.set_title('Component Signals')
            ax.set_xlabel(f'Time ({units})')
            ax.set_ylabel('Energy (eV)')
            ax.grid()
            ax.legend()
        
        fig, ax = plt.subplots()
        ax.plot(tplot,BEaplot)
        ax.set_title('BEa')
        ax.set_xlabel(f'Time ({units})')
        ax.set_ylabel('Energy (eV)')
        ax.grid()
    
    def SetEnergyParams(self,Ebounds,nsigs,SigShape):
        """ 
        Sets constraints for how to design energy program

        Args:
            Ebounds (nparray): assigns upper and lower bounds for the energy program
            nsigs (int): how many basis functions to use to construct the program
            SigShape (str): what basis function shapes to use
                *** Currently, only square is fully supported, sine is included but not guaranteed to work
        """
        
        self.Elb = Ebounds[0]
        self.Eub = Ebounds[1]
        self.SigShape = SigShape
        self.nsigs = nsigs
        self.nx = 3*self.nsigs+1
            
    def SetSignalParams(self,SigParams,printSignals=False):
        """
        Assigns parameter values to Amp, DC, Freq, and Offset for the basis functions

        Args:
            SigParams (dict or tensor): values for each signal parameter in [(Amp,DC,Freq),Offset] order
            printSignals (bool): prints the control program from these basis functions if True
        """
        if type(SigParams) == torch.tensor:
            SigParams = SigParams.numpy()
        
        if type(SigParams) != dict:
            SigParams = self.convert2dict(SigParams)
            
        self.Amp = SigParams['Amplitude']
        self.DC = SigParams['DutyCycle']
        self.Freq = SigParams['Frequency']
        self.Offset = SigParams['Offset']
        self.fmin = np.min(self.Freq)
        
        if printSignals == True:
            self.printEnergyProgram()
            
    def convert2dict(self,SigParams):
        """
        Converter function which ensures SigParams is formatted as a dictionary

        Args:
            SigParams (nparray): array of basis function parameter values
        Returns:
            tempdict (dict): SigParams formatted as a dictionary
        """
        SigParams = SigParams.reshape(-1)
        tempdict = {'Amplitude':SigParams[0:-1:3].tolist(),
           'DutyCycle':SigParams[1:-1:3].tolist(),
           'Frequency':SigParams[2:-1:3].tolist(),
           'Offset': SigParams[-1].tolist(),
           }
        return tempdict
     
    def buildModel(self):
        """
        Constructs the catalytic network model
        """
        m = self.create_pyomo_model()
        self.m=m
        
    def discModel(self,discMethod='F'):
        """ 
        Discretizes the model
        *** Although collocation is implemented, it is not supported and is not guaranteed to work
        """
        if discMethod == 'F':
            discretizer = TransformationFactory('dae.finite_difference')
            discretizer.apply_to(self.m, nfe = self.nfe, scheme='BACKWARD')
        elif discMethod == 'C':
            discretizer = TransformationFactory('dae.collocation')
            discretizer.apply_to(self.m, nfe = self.nfe, ncp=2, scheme='LAGRANGE-RADAU')
    
    def solveThermoControl(self, tee=False):
        """
        Solves for the cyclic steady-state concentrations of the network

        Args:
            tee (bool): Whether or not to print the IPOPT output text
        """
        self.m.q = 0.00083144626
        #solver = SolverFactory('ipopt')
        solver = SolverFactory('ipopt', executable = r'C:\msys64\home\jflory-admin\Ipopt\IpoptCompile3\src\Apps\AmplSolver\ipopt.exe')
        status = solver.solve(self.m, tee=tee)
    
        # Extract Concentration profiles for warm-starting square system
        self.getConcData(self.m)
    
    def solveBVP(self,nfe=250,discMethod='F',tee=False):
        """
        Wrapper function which constructs, discretizes, and solves the network model all at once

        Args:
            nfe (int): number of discretizations to use 
            discMethod ('C' or 'F'): whether to use finite difference or collocation to discretize the model
            tee (bool): If True, displays the IPOPT output text
        """
        self.nfe = nfe 
        self.buildModel()
        self.discModel(discMethod)   
        self.InitializeModel(None)
        self.solveThermoControl(tee)
    
    def calc_selectivity(self, reactant, products):
        """
        Calculates the selectivity of the desired products

        Args:
            reactant (nparray): concentration of the reactant gas species during cyclic steady-state
            products (list of nparrays): concnetrations of the desired product gas species during cyclic steady-state
        Returns:
            Sel (nparray): array of selectivity values, as a percentage, for each product species entered
        """
        n_p = len(products)
        
        r_avg = np.mean(reactant)
        
        p_avg = np.zeros(n_p)
        for i in range(n_p):
            p_avg[i] = np.mean(products[i])
        
        
        Sel = p_avg/r_avg
    
        Sel = Sel/np.sum(Sel)*100
        return Sel
        
    def calc_TOF(self, product):
        """
        Calculates the TOF of the desired product species

        Args:
            product (nparray): array of product gas concentration values during cyclic steady-state
        Returns:
            avTOF (double): average turnover frequency during a single cycle
        """
        TOF = product * value(self.m.q) / self.Nsites
        avTOF = np.mean(TOF)
        return avTOF

    def SetBOParams(self,ntrain,nnew,xL,xU):
        """ 
        Set parameter values needed to conduct Bayesian optimization loop

        Args:
            ntrain (int): how many initial points are provided before experimentation
            nnew (int): how many new experiments to conduct
            xL (list of doubles): lower bounds for the Amplitude, Duty Cycle, logscaled Frequency, and Offset parameters 
            xU (list of doubles): upper bounds for the Amplitude, Duty Cycle, logscaled Frequency, and Offset parameters 
        """
        self.ntrain = ntrain
        self.nnew = nnew
        self.xL = xL
        self.xU = xU
        
    def SetData(self,Xin,yin):
        """
        Sets data to be used in the GP model

        Args:
            Xin (tensor): tensor containing all training data inputs
            Yin (tensor): tensor contraining all training data outputs
        """
        
        self.X = Xin
        self.y = yin
    
    def GenX(self, n=None):
        """
        Generates random control programs using LatinHypercude to ensure random points are sufficiently spaced out

        Args:
            n (int): number of control programs to construct
        Returns:
            x_out (tensor): tensor containing randomly generated control program parameters
        """
        
        if n==None:
            n = self.ntrain
        # delBEa, f, DC
        sampler = qmc.LatinHypercube(d=self.nx)
        samples = torch.from_numpy(sampler.random(n=n).reshape(n,self.nx))
        x1 = samples[:,:self.nsigs]*(self.xU[0]-self.xL[0])+self.xL[0]
        x2 = x2 = samples[:,self.nsigs:2*self.nsigs]*(self.xU[1]-self.xL[1])+self.xL[1]
        x3 = 10**(samples[:,2*self.nsigs:3*self.nsigs]*(self.xU[2]-self.xL[2])+self.xL[2])
        x123 = torch.zeros([n,3*self.nsigs])
        x123[:,0::3] = x1
        x123[:,1::3] = x2
        x123[:,2::3] = x3
        x4 = samples[:,-1]*(self.xU[3]-self.xL[3])+self.xL[3]
        x_out = torch.cat([x123,x4.reshape(-1,1)],dim=1)
        return x_out
        
    def ScaleX(self, Xin=None):
        """
        Normalizes control program input values to be between 0-1 for numerical stability
        Frequency is logarithmically scaled and then normalized to be between 0-1

        Args:
            Xin (tensor): Raw input data to scale, if none is provided the function defaults to the class's X data
        Returns:
            x_out (tensor): Input data scaled to be between 0-1
        """
        if Xin == None:
            Xin = self.X
        n = len(Xin)
        x1 = (Xin[:,0:-1:3].reshape(n,-1)-self.xL[0])/(self.xU[0]-self.xL[0])
        x2 = (Xin[:,1:-1:3].reshape(n,-1)-self.xL[1])/(self.xU[1]-self.xL[1])
        x3 = (torch.log10(Xin[:,2:-1:3]).reshape(n,-1)-self.xL[2])/(self.xU[2]-self.xL[2])
        x123 = torch.zeros([n,3*self.nsigs])
        x123[:,0::3] = x1
        x123[:,1::3] = x2
        x123[:,2::3] = x3
        x4 = (Xin[:,-1].reshape(n,1)-self.xL[3])/(self.xU[3]-self.xL[3])
        x_out = torch.cat([x123,x4],dim=1)
        return x_out.to(torch.double)
    
    def RescaleX(self, Xin=None):
        """
        Takes scaled control program inputs and returns them to the physical domain

        Args:
            Xin (tensor): Input data scaled between 0-1
        Returns:
            x_out (tensor): Input data scaled back into the physcial domain
        """
        if Xin == None:
            Xin = self.X
        n = len(Xin)
        x1 = Xin[:,0:-1:3].reshape(n,-1)*(self.xU[0]-self.xL[0])+self.xL[0]
        x2 = Xin[:,1:-1:3].reshape(n,-1)*(self.xU[1]-self.xL[1])+self.xL[1]
        x3 = 10**(Xin[:,2:-1:3].reshape(n,-1)*(self.xU[2]-self.xL[2])+self.xL[2])
        x123 = torch.zeros([n,3*self.nsigs])
        x123[:,0::3] = x1
        x123[:,1::3] = x2
        x123[:,2::3] = x3
        x4 = Xin[:,-1].reshape(n,1)*(self.xU[3]-self.xL[3])+self.xL[3]
        x_out = torch.cat([x123,x4],dim=1)
        return x_out
    
    def ScaleY(self, yin=None):
        """
        Scales output data for use in the GP model
        Outputs are first logarithmically scaled and then standardized to be N(0,1)

        Args:
            yin (tensor): Raw output data to scale, if none is provided the function defaults to the class's Y data
        Returns:
            y_out (tensor): Normalized output data that has a mean of 0 and standard deviation of 1
        """
        if yin == None:
            yin = self.y
        yin = np.log10(yin)
        y_out = (yin - torch.mean(yin))/torch.std(yin)
        return y_out.to(torch.double)
    
    
        
    def trainGP(self):
        """
        Trains a GP model based on previously collected data
        """
        x_scl = self.ScaleX()
        y_scl = self.ScaleY()
        
        # Xaug, yaug = self.AugmentData(x_scl,y_scl)
        
        gp = SingleTaskGP(x_scl,y_scl)
        mll = ExactMarginalLogLikelihood(gp.likelihood,gp)
        fit_gpytorch_mll(mll)
        
        self.gp = gp
        
    def GetRec(self,af, b=2, printResults=False):
        """
        Determines next best simulation or experiment to conduct

        Args:
            af ('UCB' or 'EI'): determines which acquistion function to use
            b (double, >0): When using UCB, determines the exploration parameter value
            printResults (bool): If True, returns the AF, gp mean, and gp stdev value at the recommended point
        Returns:
            x_new (tensor): control program inputs for optimal next experiment
        """
        y_scl = self.ScaleY()
        
        if af == 'UCB':
            af = qUpperConfidenceBound(self.gp, b)
        elif af == 'EI':
            af = qExpectedImprovement(self.gp, torch.max(y_scl).item())
        x_rec, af_val = optimize_acqf(acq_function = af,
                                      bounds = torch.vstack([torch.zeros([1,self.nx]),torch.ones([1,self.nx])]),
                                      q = 1,
                                      num_restarts = 25,
                                      raw_samples = 1024)
        
        # Show mean/std/acq_value for each iteration for diagnostics
        x_mean = self.gp.posterior(x_rec).mean.detach().numpy().reshape(-1)[0]
        x_std = torch.sqrt(self.gp.posterior(x_rec).variance).detach().numpy().reshape(-1)[0]
        x_af = af_val.detach().numpy().reshape(-1)[0]
        
        # Plug in x_rec to get y_rec
        x_new = self.RescaleX(x_rec)
        
        if printResults == True:
            print(f'Mean:{x_mean:4.2f} Std:{x_std:4.2f} AF:{x_af:4.2f}') 
            
        return x_new
    
    def AddData(self,Xnew,ynew):
        """
        A shortcut function which adds new data to the class without overriding old data

        Args:
            Xnew (tensor): New input data to add to the data pool
            ynew (tensor): New output data to add to the data pool
        """
        # Add point to training data
        self.X = torch.cat((self.X,Xnew),0)
        self.y = torch.cat((self.y,torch.tensor(ynew).reshape([1,1])),0)

def importSystemFuncs():
    """
    This function takes system specific functions from the model file and adds it to the class
    """
    from System_Model import create_pyomo_model, timeParams, calc_rates, getConcData, InitializeModel
    ProCatModel.create_pyomo_model = create_pyomo_model
    ProCatModel.timeParams = timeParams
    ProCatModel.calc_rates = calc_rates
    ProCatModel.getConcData = getConcData
    ProCatModel.InitializeModel = InitializeModel