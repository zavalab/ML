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
           'LLExtraction',
           'run_NSGAII')


def aspen_files(aspen_file_path1 , aspen_file_path2):
    aspen_file_path1 = os.path.abspath(aspen_file_path1)
    aspen_file_path2 = os.path.abspath(aspen_file_path2)
    aspen1 = win32.Dispatch("Apwn.Document") # .client.Dispatch("Apwn.Document")
    aspen1.InitFromFile(aspen_file_path1)
    aspen1.Visible  = False
    aspen1.SuppressDialogs = True
    time.sleep(3)
    aspen1.Engine.Run2()
    time.sleep(3)
    """Conecta a ASPEN y carga la segunda simulacion"""
    aspen2 = win32.Dispatch("Apwn.Document") # .client.Dispatch("Apwn.Document")
    aspen2.InitFromFile(aspen_file_path2)
    aspen2.Visible  = False
    aspen2.SuppressDialogs = True
    time.sleep(1.0)
    aspen2.Engine.Run2()
    time.sleep(1.0)
    return aspen1, aspen2

class LLExtraction(ElementwiseProblem):
    
    def __init__(self, aspen1, aspen2):
        self.aspen1 = aspen1
        self.aspen2 = aspen2
        variables = dict()
        variables[f"x1"] = Real(bounds=(0.1, 3.0))   # Reflux ratio
        variables[f'x2'] = Real(bounds=(0.5, 10 ))    # Solvent ratio
        variables[f'x3'] = Integer(bounds=(0,3))    # solvent type
        variables[f'x4'] = Integer(bounds=(1,60))    # Feed stage 
        variables[f'x5'] = Integer(bounds=(20,60)) # Stages
        super().__init__(vars=variables, n_obj=2,n_ieq_constr=0)

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
            if k in ("aspen1", "aspen2"):
                # reutiliza el mismo objeto COM (no intentar copiar)
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def Error_Run(self):
        return [-0.0001, 1000] #  

    def Clean_Aspen(self):
        x = [85,2]
        #-----------------------------Column 1-----------------------------#
        N_C1 = r"\Data\Blocks\D1\Input\NSTAGE"
        FeedStage  = r"\Data\Blocks\D1\Input\FEED_STAGE\S1"  
        #Send to Aspen #
        self.aspen1.Tree.FindNode(N_C1).Value = x[0] 
        self.aspen1.Tree.FindNode(FeedStage).Value = x[1]   
        #Send to Aspen #
        self.aspen2.Tree.FindNode(N_C1).Value = x[0] 
        self.aspen2.Tree.FindNode(FeedStage).Value = x[1]   

    def Update_Aspen(self,x):
        #-----------------------------Column 1-----------------------------#
        N_C1 = r"\Data\Blocks\D1\Input\NSTAGE"
        RR_C1  = r"\Data\Blocks\D1\Input\BASIS_RR"
        solvent_flow_rate = r"\Data\Streams\SOLVENT\Input\TOTFLOW\MIXED"
        FeedStage  = r"\Data\Blocks\D1\Input\FEED_STAGE\S1"  
        #Send to Aspen #
        if x[2] == 0:
            self.aspen1.Tree.FindNode(N_C1).Value = x[4] 
            self.aspen1.Tree.FindNode(RR_C1).Value = x[0] 
            self.aspen1.Tree.FindNode(solvent_flow_rate).Value = x[1] 
            self.aspen1.Tree.FindNode(FeedStage).Value = x[3] 
        else:
            self.aspen2.Tree.FindNode(N_C1).Value = x[4] 
            self.aspen2.Tree.FindNode(RR_C1).Value = x[0] 
            self.aspen2.Tree.FindNode(solvent_flow_rate).Value = x[1] 
            self.aspen2.Tree.FindNode(FeedStage).Value = x[3] 
            if x[2] == 1:
                self.aspen2.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\1-DEC-01").Value = 1
                self.aspen2.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\1-OCT-01").Value = 0
                self.aspen2.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\N-BUT-01").Value = 0
            elif x[2] == 2:
                self.aspen2.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\1-DEC-01").Value = 0
                self.aspen2.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\1-OCT-01").Value = 1
                self.aspen2.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\N-BUT-01").Value = 0
            elif x[2] == 3:
                self.aspen2.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\1-DEC-01").Value = 0
                self.aspen2.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\1-OCT-01").Value = 0
                self.aspen2.Tree.FindNode(r"\Data\Streams\SOLVENT\Input\FLOW\MIXED\N-BUT-01").Value = 1

    def Feasibily_Check(self,x):
        Constraint_List = []
        # Restricciones de la columna extractiva
        Constraint_List.append(x[3]<x[4]) #  Feed stage solvent < NT1 
        Feasibility = all(Constraint_List) == True    # True if all constraints are True, False if one value doesnt meet the criterion
        return Feasibility

    def _evaluate(self, x, outs, *args, **kwargs ):
        """
        Ejecuta la simulación en ASPEN con los valores de las variables proporcionadas.
        
        :param variables: Lista o array de valores para las variables.
        :return: Array de resultados de los outputs.
        """
        x = np.array([x[f"x{k:01}"] for k in range(1,6)])
        # Anter de ejecutar Aspen evaluar si el punto es factible
        if self.Feasibily_Check(x) == False: # Entonces no se cumplen las restricciones del problema
            results =  self.Error_Run() 
        else:
            self.aspen1.Reinit()
            self.aspen2.Reinit()

            Run_Status_Dir = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
              # List to save ressult
            if x[2] == 0:  # El solvente uno es probado (primer simulacion)
                # if self.aspen1 is None:
                #     raise RuntimeError("ASPEN no está conectado. Llama a connect() antes de ejecutar una simulación.")
                
                # Limpia Aspen con valores default 
                self.Clean_Aspen()

                # Asigna los valores de las variables en ASPEN
                self.Update_Aspen(x)

                # Ejecuta la simulación
                self.aspen1.Engine.Run2()

                if self.aspen1.Tree.FindNode(Run_Status_Dir) == None:
                    results =  self.Error_Run()  
                else:
                    Run_Status = self.aspen1.Tree.FindNode(Run_Status_Dir).Value
                    if Run_Status == 9 or Run_Status == 10 :
                        results =  self.Error_Run() 
                        self.aspen1.Reinit()
                    else:
                        results = []
                        # PUREZa
                        pureza = self.aspen1.Tree.FindNode(r"\Data\Streams\PRODUCT\Output\MOLEFRAC\MIXED\CAPRYC").Value
                        # results.append(np.log(pureza))
                        results.append(-pureza)


                        # REBOILER  B1 #
                        REB_DUTY_D1 = r"\Data\Blocks\D1\Output\REB_DUTY"
                        ReboilerDuty = self.aspen1.Tree.FindNode(REB_DUTY_D1).Value*0.0041868    #cal/sec to kW
                        results.append(ReboilerDuty)
                        self.aspen1.Reinit()
            else: # Cualquier otro solvente
                # Limpia Aspen con valores default 
                self.Clean_Aspen()
                # Asigna los valores de las variables en ASPEN
                self.Update_Aspen(x)
                # Ejecuta la simulación
                self.aspen2.Engine.Run2()

                if self.aspen2.Tree.FindNode(Run_Status_Dir) == None:
                    results =  self.Error_Run()  
                else:
                    Run_Status = self.aspen2.Tree.FindNode(Run_Status_Dir).Value
                    if Run_Status == 9 or Run_Status == 10 :
                        results =  self.Error_Run() 
                        self.aspen2.Reinit()
                    else:
                        results = []
                        # PUREZa
                        pureza = self.aspen2.Tree.FindNode(r"\Data\Streams\PRODUCT\Output\MOLEFRAC\MIXED\CAPRYC").Value
                        # results.append(np.log(pureza))
                        results.append(-pureza)

                        # REBOILER  B1 #
                        REB_DUTY_D1 = r"\Data\Blocks\D1\Output\REB_DUTY"
                        ReboilerDuty = self.aspen2.Tree.FindNode(REB_DUTY_D1).Value*0.0041868    #cal/sec to kW
                        results.append(ReboilerDuty)
                        self.aspen2.Reinit()
        outs["F"] = [results[0], results[1]]    # Declare the functions
        outs["G"] = []

def run_NSGAII(problem, aspen1, aspen2, 
               aspen_file_path1, aspen_file_path2,
               file, iteration ):
    gen = 10 # ESTO NO IMPORTA 
    pop = 10
    ofs = 10
    termination = get_termination("n_eval", 500)
    algorithm = NSGA2(pop_size=pop, n_offsprings=ofs, sampling=MixedVariableSampling(),
                            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                            eliminate_duplicates=MixedVariableDuplicateElimination())
    res = minimize(problem,algorithm,termination,save_history=True,verbose=False) # Eliminar seed = 42 para tener diferentes resultados del algoritmo :D 
    # Tiempo de ejecucion 500 iteraciones: 21 m 12 seg
    aspen1.Quit(aspen_file_path1)
    aspen1 = None

    aspen2.Quit(aspen_file_path2)
    aspen2 = None
    sol, fun, const = [], [], []
    for i, c in enumerate(res.history):
        sol.extend(c.pop.get("F"))
        fun.extend(c.pop.get("X"))
    # print(sol)
    # print(fun)
    
    df1 = pd.DataFrame(data=sol , columns=['pureza', 'reboiler'])
    df2 = pd.DataFrame(data=fun )
    df = pd.concat([df1,df2], axis= 1)

    output_folder = 'results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    file = file + str(iteration+1) + '.csv'
    output_path = os.path.join(output_folder, file)
    df.to_csv( output_path , index=False)  
    # df.to_csv(r'LLExtraction_NSGAII.csv', index=False)  


















