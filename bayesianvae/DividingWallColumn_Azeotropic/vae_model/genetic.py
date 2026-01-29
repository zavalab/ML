# Utilities
import os                          # Import operating system interface
import win32com.client as win32    # Import COM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import dill
import seaborn as sns
import time 
import copy

from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
from tqdm.notebook import tqdm

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['figure.figsize'] = [7, 5]  # Largo, ancho  
plt.rcParams["text.usetex"] = False
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
import seaborn as sns
#import dill
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Binary
from pymoo.algorithms.moo.nsga2 import NSGA2 #, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.termination import get_termination
from pymoo.termination.max_gen import MaximumGenerationTermination


__all__ = ('aspen_files',
           'Intensified',
           'run_NSGAII')


def aspen_files(aspen_file_path1):
    aspen_file_path1 = os.path.abspath(aspen_file_path1)
    aspen1 = win32.Dispatch("Apwn.Document") # .client.Dispatch("Apwn.Document")
    aspen1.InitFromFile(aspen_file_path1)
    aspen1.Visible  = False
    aspen1.SuppressDialogs = True
    time.sleep(1.5)
    aspen1.Engine.Run2()
    time.sleep(1.5)
    return aspen1


class Intensified(ElementwiseProblem):
    
    def __init__(self, aspen):
        self.aspen = aspen

        variables = dict()
        variables[f"x1"] = Real(bounds=(2.0, 5.0))   # Reflux ratio
        variables[f'x2'] = Real(bounds=(50, 100 ))    # Solvent ratio
        variables[f'x3'] = Real(bounds=(50, 60 ))    # Vapor flowrate
        variables[f'x4'] = Integer(bounds=(1,60))    # Feed stage
        variables[f'x5'] = Integer(bounds=(1,60))    # Feed stage 
        variables[f'x6'] = Integer(bounds=(1,60)) # Stages
        variables[f'x7'] = Integer(bounds=(30,50)) # Stages
        super().__init__(vars=variables, n_obj=2, n_ieq_constr=0)

    def __deepcopy__(self, memo):
        """
        Evita copiar los objetos COM (aspen1, aspen2), que no son 'picklable'.
        Solo se copian profundamente los demás atributos y se reutilizan
        las mismas referencias a ASPEN.
        """
        cls = self.__class__
        # crear instancia vacía sin llamar __init__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k in ("aspen"):
                # reutiliza el mismo objeto COM (no intentar copiar)
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def Column(self, nt1,  ds , P, payback = 3,
        M_S = 1716,fm = 3.67, ft = 0):
        # Inputs
        # M_S : M&S is Marshall and Swift Equipment Cost Index year 2019
        # fm value for stainless steel
        # d : divided column diameter  in meters
        # nt : number of trays
        # ft : tray sieve
        # Variables
        # L : column length in meters  
        H = 1.2*0.61*(nt1-2)
        fp = 1 + 0.00074*(P - 3.48) + 0.00023*(P-3.48)**2
        fc = fm*fp
        CostColumn =  (M_S/280)*(957.9*(ds**1.066)*(H**0.802))*(2.18 + fc) 
        PlatosColumn =  (M_S/280)*(round(nt1))*97.2*(ds**1.55)*(ft + fm)
        TotalCost = (CostColumn + PlatosColumn)/payback   # [USD/yr]
        return TotalCost

    def CondenserColumn(self,COND_DUTY,TOP_TEMP, payback = 3, labour = 8000, M_S = 1716, \
        ktc =  0.852,fd = 0.8,fm =2.81,fp = 0 ):
        # Inputs
        # M_S : M&S is Marshall and Swift Equipment Cost Index year 2019
        # ktc : Heater-transfer coefficient  in kW/m2K
        # Fm  : carbon steel shell, stainless steel tube
        # Fd  : fixed tube sheet
        # Fp  : pressures lower than 10.3 bar 
        # --------------------------#
        # Accessing values of ASPEN #
        # --------------------------#
        Condenserduty = self.aspen.Tree.FindNode(COND_DUTY).Value*0.0041868  # cal/sec to kW
        Tcond  = self.aspen.Tree.FindNode(TOP_TEMP).Value + 273.15 #Kelvyn
        # --------------------------#
        CoolingService = [278, 310]  # [K] Chilled water, cooling water # 5 y 36.85°C
        Utilitiprice = [4.43, 0.354] # [$/GJ]
        # TcoolingSErvice < Tcond ...
        if CoolingService[1] < Tcond:  
            # Use cooling water
            Tcools = CoolingService[1]
            Uprice = Utilitiprice[1]
        elif CoolingService[0] < Tcond:
            # Use chilled water
            Tcools = CoolingService[0]
            Uprice = Utilitiprice[0]
        # ------------ #
        # Capital Cost #
        # ------------ #
        DeltaT_Condenser = Tcond - Tcools 
        AreaCondenser = abs(Condenserduty)/(ktc*(DeltaT_Condenser)) # [m2]
        fc = (fd+fp)*fm
        K = M_S*1.695*(2.29 + fc)
        CostCondenser = K*(AreaCondenser**0.65)
        # --------------- #
        # Utilititie Cost #
        # --------------- #     
        UtiCondenser = Uprice*abs(Condenserduty)*(1e-6)*labour*3600  # [$/year]        
        TotalCost = CostCondenser/payback + UtiCondenser  # [$/year]
        return TotalCost

    def ReboilerColumn(self,REB_DUTY,BOTTOM_TEMP, payback = 3, labour = 8000, M_S = 1716, \
        ktc =  0.568,fd =1.35,fm=2.81, fp = 0):
        # Inputs
        # BOTTOM_TEMP, REB_DUTY : direction of parameters in ASPEN 
        # payback, labour : payback period and labour hours
        # M_S : M&S is Marshall and Swift Equipment Cost Index year 2019
        # ktc : Heater-transfer coefficient  in kW/m2K
        # Fd  : fixed tube sheet
        # Fm  : carbon steel shell, stainless steel tube
        # Fp_bar  : pressure of condenser of 1 bar (default value)
        # --------------------------#
        # Accessing values of ASPEN #
        # --------------------------#
        Temp  = self.aspen.Tree.FindNode(BOTTOM_TEMP).Value + 273.15 # Celsius to Kelvyn
        ReboilerDuty = self.aspen.Tree.FindNode(REB_DUTY).Value*0.0041868    #cal/sec to kW
        ReboilerService = [433,457,527]  # Tempearture [K]
        Utilitiprice = [7.78,8.22,9.8]     # [$/GJ]
        #LatenHeatService = [2085.03,1998.55,1697.79] #KJ/Kg
        if Temp < ReboilerService[0]:       #Use low pressure steam 
            #L = LatenHeatService[0]
            Uprice = Utilitiprice[0]
            Treb = ReboilerService[0]
        elif Temp < ReboilerService[1]:     #Use medium pressure steam
            #L = LatenHeatService[1]
            Uprice = Utilitiprice[1]
            Treb = ReboilerService[1]
        elif Temp < ReboilerService[2]:     #Use high pressure steam
            #L = LatenHeatService[2]
            Uprice = Utilitiprice[2]
            Treb = ReboilerService[2]

        # ------------ #
        # Capital Cost #
        # ------------ #
        fc = (fd + fp)*fm
        DeltaT_Reboiler = Treb - Temp   # Steam temperature-base temperature
        AreaReboiler = ReboilerDuty/(DeltaT_Reboiler*ktc)
        K = M_S*1.695*(2.29 + fc)
        CostReboiler = K*(AreaReboiler**0.65)  #[$]
        # --------------- #
        # Utilititie Cost #
        # --------------- #
        UtiReboiler = Uprice*ReboilerDuty*(1e-6)*labour*3600  # [$/year]
        TotalCost = CostReboiler/payback + UtiReboiler  # [$/year]
        return TotalCost

    def Antoine_function(self,Antoine1,Antoine2,Antoine3,Antoine4,Antoine5,Antoine6,Antoine7,T):
        ListPsat = [] #np.zeros(componentes)
        for i in range(len(Antoine1)):
            ListPsat.append(np.exp(Antoine1[i] + Antoine2[i]/(T+Antoine3[i]) + Antoine4[i]*T + Antoine5[i]*np.log(T) +  Antoine6[i]*T**Antoine7[i]))
        return np.array(ListPsat)

    def Antoine_values(self, listofcomponents):
        # Parameters for vapor pressure (Extended Antoine)
        listofvalues = ['VAL1','VAL2','VAL3','VAL4','VAL5','VAL6','VAL7']  #Default values given in Aspen
        listA, listB, listC, listD, listE, listF, listG = [],[],[],[],[],[],[]
        for i in listofvalues:
            for j in listofcomponents:
                path_constants = r"\Data\Properties\Parameters\Pure Components\PLXANT-1\Input\{value}\PLXANT\{comp}".format(value= i, comp = j)
                if i == 'VAL1':
                    listA.append(float(self.aspen.Tree.FindNode(path_constants).Value))
                elif i == 'VAL2':
                    listB.append(float(self.aspen.Tree.FindNode(path_constants).Value))
                elif i == 'VAL3':
                    listC.append(float(self.aspen.Tree.FindNode(path_constants).Value))
                elif i == 'VAL4':
                    listD.append(float(self.aspen.Tree.FindNode(path_constants).Value))
                elif i == 'VAL5':
                    listE.append(float(self.aspen.Tree.FindNode(path_constants).Value))
                elif i == 'VAL6':
                    listF.append(float(self.aspen.Tree.FindNode(path_constants).Value))
                elif i == 'VAL7':
                    listG.append(float(self.aspen.Tree.FindNode(path_constants).Value))  
        return listA, listB, listC, listD, listE, listF, listG

    def EnthalpyCombustion(self, listofcomponents):
        # Enthalpys of combustion from aspen in cal/mol
        listHcom = []
        for i in listofcomponents:
            dic = r"\Data\Properties\Parameters\Pure Components\REVIEW-1\Input\VALUE\HCOM\{comp}"
            Hcom = float(self.aspen.Tree.FindNode(dic.format(comp = i)).Value)/1000  #cal/mol a kcal/mol 
            Hcom = Hcom*4184  #J/mol
            listHcom.append(Hcom) 
        return  np.array(listHcom)

    def variables_FEDI(self,listofcomponents, nt1):
        x_in1 , x_in2, x_ins, x_in3, Ts = [] , [], [], [] , []
        Feed_1_dir = r"\Data\Streams\FEED\Input\FLOW\MIXED\{comp}" 
        Feed_2_dir = r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\{comp}"
        for i in listofcomponents:
            # Extract molar composition information of stream 1
            if self.aspen.Tree.FindNode(Feed_1_dir.format(comp = i)).Value == None:
                x_in1.append(0.00)
            else:
                x_in1.append(self.aspen.Tree.FindNode(Feed_1_dir.format(comp = i)).Value)
            # Extract molar composition information of stream 2
            if self.aspen.Tree.FindNode(Feed_2_dir.format(comp = i)).Value == None:
                x_in2.append(0.00)
            else:
                x_in2.append(self.aspen.Tree.FindNode(Feed_2_dir.format(comp = i)).Value) 
            # Bottom streams form column 1 entering in column 2 
            x_in3.append(self.aspen.Tree.FindNode(r"\Data\Blocks\B1\Output\X\{nts}\{comp}".format(nts = str(int(nt1)), comp = str(i) )).Value)
        # Normalize the composition of both streams entering in the column 
        F1 = self.aspen.Tree.FindNode(r"\Data\Streams\FEED\Input\TOTFLOW\MIXED").Value
        F2 = self.aspen.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED").Value
        T_in_1 = self.aspen.Tree.FindNode(r"\Data\Streams\5\Output\TEMP_OUT\MIXED").Value + 273.15 #[K]
        T_in_2 = self.aspen.Tree.FindNode(r"\Data\Streams\3\Output\TEMP_OUT\MIXED").Value + 273.15 #[K]
        x_ins.append( ((np.array(F1)*np.array(x_in1)   +   np.array(F2)*np.array(x_in2)) / (F1+F2)) )
        #Mean Temperature between solvent and azeotropic mixture
        Ts.append((T_in_1+T_in_2)/2) 
        # Temperature of bottom of first column
        Ts.append(self.aspen.Tree.FindNode(r"\Data\Blocks\{block}\Output\BOTTOM_TEMP".format(block = 'B1')).Value + 273.15 ) 
        # Composition of bottom stream 
        x_ins.append(np.array(x_in3))
        return x_ins,  Ts
    
    def Fedi(self, tags, listofcomponents, NF, NR, x_ins, Ts, d, nt, FlashPoint_val, IgnitionTemperature_val):
        # Enthalpys of combustion from aspen in cal/mol
        listHcom = self.EnthalpyCombustion(listofcomponents)
        # Parameters for vapor pressure (Extended Antoine)
        listA, listB, listC, listD, listE, listF, listG = self.Antoine_values(listofcomponents)
        # Calculations of fedi for each column
        Fedi_list = []
        for i in range(len(tags)):
            x_in = x_ins[i]
            block_tag = tags[i]
            T  = Ts[i]
            FlashPoint = sum(FlashPoint_val*x_in)
            AutoIgnitionTemp = sum(IgnitionTemperature_val*x_in)
            Vol = (np.pi * ( d/2)**2)*(1.2*0.61*(nt-2))  # [m3]
            Pcolumn = float(self.aspen.Tree.FindNode(r"\Data\Blocks\{block}\Input\PRES1".format(block=block_tag)).Value*101.325)  # [atm a kPa]
            Mass =  float(self.aspen.Tree.FindNode(r"\Data\Blocks\{block}\Output\BAL_MASI_TFL".format(block=block_tag)).Value/3600)  # [Kg/sec]
            Treb = self.aspen.Tree.FindNode(r"\Data\Blocks\{block}\Output\BOTTOM_TEMP".format(block=block_tag)).Value + 273.15 # Celsius to Kelvyn
            #---------------F1,F2,F3,F4-----------------#
            Entalpy_comb = sum(abs(listHcom)*x_in )  # [KJ/mol] 
            # Note: de acuerdo con el paper son J/mol pero da valores muy altos de FEDI en todas las unidades
            
            F1 = 0.1*(Mass*(Entalpy_comb))/3.148
            F2 = (6/3.148)*Pcolumn*Vol
            Psat_in = self.Antoine_function(listA,listB, listC, listD, listE, listF, listG, T)*100 # [Bar a kPa]
            VapPress = sum(Psat_in*x_in )
            F3 = (1e-3)*(1/Treb)*((Pcolumn-VapPress)**2)*Vol   # La temperatura es de la operación
            # Penalty 1
            if T > FlashPoint and T < 0.75*AutoIgnitionTemp: 
                pn1 = (1.45 +  1.75)/2
            elif T > 0.75*AutoIgnitionTemp:
                pn1 = 1.95
            else:
                pn1 = 1.1
            # Penalty 2
            if VapPress > 101.325 and Pcolumn > VapPress :
                pn2 = 1 + (Pcolumn-VapPress)*0.6/Pcolumn
                F = F2 + F3
            else:
                pn2 = 1 + (Pcolumn-VapPress)*0.4/Pcolumn
                F = F2
            if VapPress < 101.325 and 101.325 < Pcolumn:
                pn2 = 1 + (Pcolumn-VapPress)*0.2/Pcolumn
                F = F3
            else:
                pn2 = 1.1
                F = F3     
            # Penalty 4
            pn4 = 1 + 0.25*(np.array(NF).max() + np.array(NR).max())
            pn3,pn5,pn6 = 1,1,1
            Damage_Potential = (F1*pn1 + F*pn2)*(pn3*pn4*pn5*pn6)
            Fedi_list.append(4.76*(Damage_Potential**(1/3)) )
        return Fedi_list 

    def CO2(self,tags, NHV = 39771 , C = 86.5, alfa = 3.67):
        CO2 = []
        for i in range(len(tags)):
            Temp  = self.aspen.Tree.FindNode(r"\Data\Blocks\{block}\Output\BOTTOM_TEMP".format(block = tags[i])).Value + 273.15 # Celsius to Kelvyn
            ReboilerDuty = self.aspen.Tree.FindNode(r"\Data\Blocks\{block}\Output\REB_DUTY".format(block = tags[i])).Value*0.0041868    #cal/sec to kW
            ReboilerService = [433,457,527]  # Tempearture [K]
            LatenHeatService = [2085.03,1998.55,1697.79]    #KJ/Kg
            EnthalpyService = [2758.65,2780.06,2802.23]     #KJ/Kg
            if Temp < ReboilerService[0]:       #Use low pressure steam 
                L = LatenHeatService[0]
                Enthalpy = EnthalpyService[0]
            elif Temp < ReboilerService[1]:     #Use medium pressure steam
                L = LatenHeatService[1]
                Enthalpy = EnthalpyService[1]
            elif Temp < ReboilerService[2]:     #Use high pressure steam
                L = LatenHeatService[2]
                Enthalpy = EnthalpyService[2]
            #The boiler feed water is assumed to be at 100 °C with an enthalpy of 419 kJ/kg
            Tftb = 1800 + 273.15    #[K]
            Tstack = 160 + 273.15   #[K]
            To = 25+273.15          #[K]
            efficiency = (Tftb - To)/(Tftb - Tstack)
            Q_Fuel = (ReboilerDuty/L)*(Enthalpy - 419)*efficiency
            CO2_val = (Q_Fuel/NHV)*(C/100)*alfa
            CO2.append(CO2_val) # [kg/hr]
        return sum(np.array(CO2))*8760   # [kg/yr]
 
    def Feasibily_Check(self, x):
        Constraint_List = []
        Constraint_List.append(x[3]<x[4]) #  Feed stage solvent < Feed stage azeotropic 
        Constraint_List.append(x[3]<x[5]) #  Feed stage azeotropic < NTDW 
        Constraint_List.append(x[4]<x[5]) # Feed stage solvent < NTDW
        Constraint_List.append(x[5]< (x[6]-3)) # NTDWC < NTCOL
        Feasibility = all(Constraint_List) == True    # True if all constraints are True, False if one value doesnt meet the criterion
        return Feasibility

    def Update_Aspen(self,x):
        #-----------------------------Column 1-----------------------------#
        N_C1 = r"\Data\Blocks\B1\Input\NSTAGE"
        RR_C1  = r"\Data\Blocks\B1\Input\BASIS_RR"
        solvent_flow_rate = r"\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED"
        FeedStage_1  = r"\Data\Blocks\B1\Input\FEED_STAGE\3"  
        FeedStage_2 = r"\Data\Blocks\B1\Input\FEED_STAGE\5"
        vapor_flow = r"\Data\Blocks\B1\Input\PROD_FLOW\VR"
        #Send to Aspen #
        self.aspen.Tree.FindNode(N_C1).Value = x[6] 
        self.aspen.Tree.FindNode(RR_C1).Value = x[0] 
        self.aspen.Tree.FindNode(solvent_flow_rate).Value = x[1] 
        self.aspen.Tree.FindNode(FeedStage_1).Value = x[3]   
        self.aspen.Tree.FindNode(FeedStage_2).Value = x[4]
        self.aspen.Tree.FindNode(vapor_flow).Value = x[2]
        # Internals of column 1
        B1_INT_1 = r"\Data\Blocks\B1\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\INT"
        self.aspen.Tree.FindNode(B1_INT_1).Value = x[6] - 1  # Actualizar internals de primera columna


        #-----------------------------Column 2-----------------------------#
        N_C2 = r"\Data\Blocks\B2\Input\NSTAGE"
        #Send to Aspen #
        self.aspen.Tree.FindNode(N_C2).Value = x[5] # Se alimenta en el fondo de la columna o NT2 
        vapor_source_stage  = r"\Data\Blocks\B1\Input\PROD_STAGE\VR"
        self.aspen.Tree.FindNode(vapor_source_stage).Value = x[5]+1  # A stage below
        liquid_feedstage = r"\Data\Blocks\B1\Input\FEED_STAGE\S1"
        self.aspen.Tree.FindNode(liquid_feedstage).Value = x[5]+1  # A stage below
        vapor_feedstage = r"\Data\Blocks\B2\Input\FEED_STAGE\VR"
        self.aspen.Tree.FindNode(vapor_feedstage).Value = x[5]   # In the final stage
        liquid_product  = r"\Data\Blocks\B2\Input\PROD_STAGE\S1"
        self.aspen.Tree.FindNode(liquid_product).Value = x[5]   # In the final stage
        # Internals of column 2
        B2_INT_1 = r"\Data\Blocks\B2\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\INT"
        self.aspen.Tree.FindNode(B2_INT_1).Value = x[5]   # Actualizar internals de segunda columna

    def Clean_Aspen(self):
        #-----------------------------Column 1-----------------------------#
        N_C1 = r"\Data\Blocks\B1\Input\NSTAGE"
        FeedStage_1  = r"\Data\Blocks\B1\Input\FEED_STAGE\3"  
        FeedStage_2 = r"\Data\Blocks\B1\Input\FEED_STAGE\5"
        #Send to Aspen #
        self.aspen.Tree.FindNode(N_C1).Value = 60
        self.aspen.Tree.FindNode(FeedStage_1).Value = 1  
        self.aspen.Tree.FindNode(FeedStage_2).Value = 2
        # Internals of column 1
        B1_INT_1 = r"\Data\Blocks\B1\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\INT"
        self.aspen.Tree.FindNode(B1_INT_1).Value = 60 - 1  # Actualizar internals de primera columna


        #-----------------------------Column 2-----------------------------#
        N_C2 = r"\Data\Blocks\B2\Input\NSTAGE"
        #Send to Aspen #
        self.aspen.Tree.FindNode(N_C2).Value = 30 # Se alimenta en el fondo de la columna o NT2 
        vapor_source_stage  = r"\Data\Blocks\B1\Input\PROD_STAGE\VR"
        self.aspen.Tree.FindNode(vapor_source_stage).Value = 30+1  # A stage below
        liquid_feedstage = r"\Data\Blocks\B1\Input\FEED_STAGE\S1"
        self.aspen.Tree.FindNode(liquid_feedstage).Value = 30+1  # A stage below
        vapor_feedstage = r"\Data\Blocks\B2\Input\FEED_STAGE\VR"
        self.aspen.Tree.FindNode(vapor_feedstage).Value = 30   # In the final stage
        liquid_product  = r"\Data\Blocks\B2\Input\PROD_STAGE\S1"
        self.aspen.Tree.FindNode(liquid_product).Value = 30   # In the final stage
        # Internals of column 2
        B2_INT_1 = r"\Data\Blocks\B2\Subobjects\Column Internals\INT-1\Input\CA_STAGE2\INT-1\INT"
        self.aspen.Tree.FindNode(B2_INT_1).Value = 30   # Actualizar internals de segunda columna

    def Error_Run(self):
        return [-0.001, 10]

    def B1_Check(self,listofcomponents, nt1 ):
        x_in_check = []
        for i in listofcomponents:
            x_in_check.append(self.aspen.Tree.FindNode(r"\Data\Blocks\B1\Output\X\{nts}\{comp}".format(nts = str(int(nt1)), comp = str(i) )) == None)
        return x_in_check
    
    def B2_Check(self,listofcomponents, nt2 ):
        x_in_check = []
        for i in listofcomponents:
            x_in_check.append(self.aspen.Tree.FindNode(r"\Data\Blocks\B2\Output\X\{nts}\{comp}".format(nts = str(int(nt2)), comp = str(i) )) == None)
        return x_in_check

    def _evaluate(self, x, outs, *args, **kwargs ):
        """
        Ejecuta la simulación en ASPEN con los valores de las variables proporcionadas.
        
        :param variables: Lista o array de valores para las variables.
        :return: Array de resultados de los outputs.
        """
        #----------------------Constraints function-------------#
        x = np.array([x[f"x{k:01}"] for k in range(1,8)])
        # Anter de ejecutar Aspen evaluar si el punto es factible
        output_nodes = [r"\Data\Streams\DCM\Output\MOLEFRAC\MIXED\DICHL-01"]       
        listofcomponents = ['METHA-01','DICHL-01','N:N-D-01']
        if self.Feasibily_Check(x) == False: # Entonces no se cumplen las restricciones del problema
            results =  self.Error_Run() 
        else:
        
            # Limpia Aspen con valores default   
            self.Clean_Aspen()
            # Asigna los valores de las variables en ASPEN
            self.Update_Aspen(x)
            #-Run problem -#
            self.aspen.Engine.Run2()
            
            Run_Status_Dir = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
            #First Check if Status return a value
            if self.aspen.Tree.FindNode(Run_Status_Dir) == None:  # This means there was a problem
                results = self.Error_Run()    
            else:
                Run_Status = self.aspen.Tree.FindNode(Run_Status_Dir).Value  
                B1Check= self.B1_Check(listofcomponents, x[6])
                B2Check= self.B2_Check(listofcomponents, x[5])
                if Run_Status == 9 or Run_Status == 10 or all(B1Check) == True or all(B2Check) == True: # Error in the simulation 
                    results =  self.Error_Run() 
                    self.aspen.Reinit()
                else: # Simulation without errors
                    # print('pasa punto')
                    results = []
                    #  Columns Cost  #
                    nt1, nt2 = x[6].item(), x[5].item()               # Value of number of stages 

                    d1 = self.aspen.Tree.FindNode(r"\Data\Blocks\B1\Subobjects\Column Internals\INT-1\Subobjects\Sections\INT\Input\CA_DIAM\INT-1\INT").Value
                    d2 = self.aspen.Tree.FindNode(r"\Data\Blocks\B2\Subobjects\Column Internals\INT-1\Subobjects\Sections\INT\Input\CA_DIAM\INT-1\INT").Value
                            
                    P1 = self.aspen.Tree.FindNode(r"\Data\Blocks\B1\Input\PRES1").Value * 1.01325
                    P2 = self.aspen.Tree.FindNode(r"\Data\Blocks\B1\Input\PRES1").Value * 1.01325   # Pressure in bars       
                    CostColumn =  self.Column(nt1=nt1,ds = max(d1,d2) , P = P1 )  # [USD/Yr]
                    # COST OF CONDENSERS #
                    COND_DUTY = r"\Data\Blocks\B1\Output\COND_DUTY"
                    TOP_TEMP = r"\Data\Blocks\B1\Output\TOP_TEMP"
                    CostCondenser = self.CondenserColumn(COND_DUTY,TOP_TEMP) # [USD/Yr]
                        
                    #--------------#
                    # REBOILER  B1 #
                    #--------------#
                    REB_DUTY = r"\Data\Blocks\B1\Output\REB_DUTY"
                    BOTTOM_TEMP = r"\Data\Blocks\B1\Output\BOTTOM_TEMP"
                    CostReboiler = self.ReboilerColumn(REB_DUTY,BOTTOM_TEMP) # [USD/Yr]
                    TAC1  = (CostColumn + CostCondenser +  CostReboiler)/1e6  
                    #----------------#
                    # Rectificador B2#
                    # CONDENSER   B2 #
                    #----------------#
                    COND_DUTY = r"\Data\Blocks\B2\Output\COND_DUTY"
                    TOP_TEMP = r"\Data\Blocks\B2\Output\TOP_TEMP"
                    CostCondenser = self.CondenserColumn(COND_DUTY,TOP_TEMP) # [USD/Yr]
                    TAC2  = (CostCondenser)/1e6  
                    TAC = TAC1 + TAC2
                    
                    c1_mole_frac = r"\Data\Streams\DCM\Output\MOLEFRAC\MIXED\DICHL-01"
                    c2_mole_frac = r"\Data\Streams\MEOH\Output\MOLEFRAC\MIXED\METHA-01"
                    c3_mole_frac = r"\Data\Streams\DMF\Output\MOLEFRAC\MIXED\N:N-D-01"
                    XD1  = self.aspen.Tree.FindNode(c1_mole_frac).Value
                    # XD2  = self.aspen.Tree.FindNode(c2_mole_frac).Value 
                    # XD3  = self.aspen.Tree.FindNode(c3_mole_frac).Value 

                    results.append(-XD1)
                    

                    listofcomponents = ['METHA-01','DICHL-01','N:N-D-01']
                    FlashPoint_val = np.array([9 , -4 , 58]) +273.15                 # [K] 
                    IgnitionTemperature_val = np.array([ 464, 605, 440]) + 273.15    # [K]
                    NF, NR = np.array([3,1,2]),np.array([0,0,0])                     # NFPA values
                    x_ins, Ts = self.variables_FEDI(listofcomponents,nt1=nt1)
                    Fedi = self.Fedi(['B1'], listofcomponents, NF, NR, x_ins, Ts, max(d1,d2), nt1, FlashPoint_val, IgnitionTemperature_val)
                    results.append(Fedi[0])
                    # results.append(TAC)
        outs["F"] = [results[0], results[1]]    # Declare the functions
        outs["G"] = []
     

def run_NSGAII(problem, aspen1, 
               aspen_file_path1,
               file, iteration ):
    gen = 10 # ESTO NO IMPORTA 
    pop = 50
    ofs = 10
    termination = get_termination("n_eval", 500)
    algorithm = NSGA2(pop_size=pop, n_offsprings=ofs, sampling=MixedVariableSampling(),
                            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                            eliminate_duplicates=MixedVariableDuplicateElimination())
    res = minimize(problem,algorithm,termination,save_history=True,verbose=False) # Eliminar seed = 42 para tener diferentes resultados del algoritmo :D 
    # Tiempo de ejecucion 500 iteraciones: 21 m 12 seg
    aspen1.Quit(aspen_file_path1)
    aspen1 = None

    sol, fun, const = [], [], []
    for i, c in enumerate(res.history):
        sol.extend(c.pop.get("F"))
        fun.extend(c.pop.get("X"))
    
    df1 = pd.DataFrame(data=sol , columns=['pureza', 'fedi'])
    df2 = pd.DataFrame(data=fun )
    df = pd.concat([df1,df2], axis= 1)
    df = df.drop_duplicates()

    output_folder = 'results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    file = file + str(iteration+1) + '.csv'
    output_path = os.path.join(output_folder, file)
    df.to_csv( output_path , index=False)  


