import win32com.client as win32 
import os                          # Import operating system interface
import time
import numpy as np

__all__ = ('AspenSimulation_Purity',
           'AspenSimulation_Reboiler',
           'AspenSimulation_MOBO')

# This class is adapted to work only with two simulations and to return purity

class AspenSimulation_Purity:
    
    def __init__(self, aspen_file_path1, aspen_file_path2):
        """
        Initializes the connection to ASPEN and sets the paths for variables and outputs.

        :param aspen_file_path: Path to the ASPEN file.
        :param variable_nodes: List of paths for variables in ASPEN.
        :param output_nodes: List of paths for outputs in ASPEN.
        """
        self.aspen_file_path1 = os.path.abspath(aspen_file_path1)
        self.aspen1 = None

        self.aspen_file_path2 = os.path.abspath(aspen_file_path2)
        self.aspen2 = None

    def connect(self):
        import win32com.client as win32 
        """Connect to ASPEN and load the first simulation"""
        self.aspen1 = win32.Dispatch("Apwn.Document") # .client.Dispatch("Apwn.Document")
        self.aspen1.InitFromFile(self.aspen_file_path1)
        self.aspen1.Visible  = False
        self.aspen1.SuppressDialogs = True
        time.sleep(3)
        self.aspen1.Engine.Run2()
        time.sleep(3)
        """Connect to ASPEN and load the second simulation"""
        self.aspen2 = win32.Dispatch("Apwn.Document") # .client.Dispatch("Apwn.Document")
        self.aspen2.InitFromFile(self.aspen_file_path2)
        self.aspen2.Visible  = False
        self.aspen2.SuppressDialogs = True
        time.sleep(1.1)
        self.aspen2.Engine.Run2()
        time.sleep(1.1)

    def Error_Run(self):
        return [0.0001] #  

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
        Constraint_List.append(x[3]<x[4])  
        Feasibility = all(Constraint_List) == True   
        return Feasibility

    def run_simulation(self, variables):
        """
        Runs the simulation in ASPEN with the values ​​of the provided variables.
        :param variables: List or array of values ​​for the variables.
        :return: Array of output results.
        """
        self.aspen1.Reinit()
        self.aspen2.Reinit()
        # Before running Aspen evaluate if the point is feasible
        if self.Feasibily_Check(variables) == False:
            print("Unfeasible point tested, point is penalized")
            results =  self.Error_Run() 
            return results 
        Run_Status_Dir = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
        results = []  # List to save ressult
        if variables[2] == 0:  # Solvent one is tested (first simulation)
            if self.aspen1 is None:
                raise RuntimeError("ASPEN is not connected. Call connect() before running a simulation.")
            
            # Clean Aspen with default values
            self.Clean_Aspen()

            # Assigns the values ​​of the variables in ASPEN
            self.Update_Aspen(variables)

            # Run the simulation
            self.aspen1.Engine.Run2()

            if self.aspen1.Tree.FindNode(Run_Status_Dir) == None:
                results =  self.Error_Run()  
            else:
                Run_Status = self.aspen1.Tree.FindNode(Run_Status_Dir).Value
                if Run_Status == 9 or Run_Status == 10:
                    results =  self.Error_Run() 
                    self.aspen1.Reinit()
                else:
                    # Pureza
                    pureza = self.aspen1.Tree.FindNode(r"\Data\Streams\PRODUCT\Output\MOLEFRAC\MIXED\CAPRYC").Value
                    results.append(pureza)


                    # REBOILER  B1 #
                    # REB_DUTY_D1 = r"\Data\Blocks\D1\Output\REB_DUTY"
                    # ReboilerDuty = self.aspen1.Tree.FindNode(REB_DUTY_D1).Value*0.0041868    #cal/sec to kW
                    #results.append(-1*ReboilerDuty)
                    self.aspen1.Reinit()
        else:  # Other solvent
            if self.aspen2 is None:
                raise RuntimeError("ASPEN is not connected. Call connect() before running a simulation.")
            # Clean Aspen 
            self.Clean_Aspen()
            # Update Aspen
            self.Update_Aspen(variables)
            # Run simulation
            self.aspen2.Engine.Run2()

            if self.aspen2.Tree.FindNode(Run_Status_Dir) == None:
                results =  self.Error_Run()  
            else:
                Run_Status = self.aspen2.Tree.FindNode(Run_Status_Dir).Value
                if Run_Status == 9 or Run_Status == 10:
                    results =  self.Error_Run() 
                    self.aspen2.Reinit()
                else:
                    # PUREZa
                    pureza = self.aspen2.Tree.FindNode(r"\Data\Streams\PRODUCT\Output\MOLEFRAC\MIXED\CAPRYC").Value
                    # results.append(np.log(pureza))
                    results.append(pureza)

                    # REBOILER  B1 #
                    REB_DUTY_D1 = r"\Data\Blocks\D1\Output\REB_DUTY"
                    ReboilerDuty = self.aspen2.Tree.FindNode(REB_DUTY_D1).Value*0.0041868    #cal/sec to kW
                    #results.append(-1*ReboilerDuty)
                    self.aspen2.Reinit()

        return results

    def close(self):
        """Close the connection to ASPEN."""
        if self.aspen1 is not None:
            self.aspen1.Close()
            self.aspen1 = None
        if self.aspen2 is not None:
            self.aspen2.Close()
            self.aspen2 = None


class AspenSimulation_Reboiler:
    
    def __init__(self, aspen_file_path1, aspen_file_path2):
        """
        Initializes the connection to ASPEN and sets the paths for variables and outputs.

        :param aspen_file_path: Path to the ASPEN file.
        :param variable_nodes: List of paths for variables in ASPEN.
        :param output_nodes: List of paths for outputs in ASPEN.
        """
        self.aspen_file_path1 = os.path.abspath(aspen_file_path1)
        self.aspen1 = None

        self.aspen_file_path2 = os.path.abspath(aspen_file_path2)
        self.aspen2 = None

    def connect(self):
        import win32com.client as win32 
        """Connect to ASPEN and load the first simulation"""
        self.aspen1 = win32.Dispatch("Apwn.Document") # .client.Dispatch("Apwn.Document")
        self.aspen1.InitFromFile(self.aspen_file_path1)
        self.aspen1.Visible  = False
        self.aspen1.SuppressDialogs = True
        time.sleep(3)
        self.aspen1.Engine.Run2()
        time.sleep(3)
        """Connect to ASPEN and load the second simulation"""
        self.aspen2 = win32.Dispatch("Apwn.Document") # .client.Dispatch("Apwn.Document")
        self.aspen2.InitFromFile(self.aspen_file_path2)
        self.aspen2.Visible  = False
        self.aspen2.SuppressDialogs = True
        time.sleep(1.1)
        self.aspen2.Engine.Run2()
        time.sleep(1.1)

    def Error_Run(self):
        return [-1000] #  

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
        Constraint_List.append(x[3]<x[4])  
        Feasibility = all(Constraint_List) == True   
        return Feasibility

    def run_simulation(self, variables):
        """
        Runs the simulation in ASPEN with the values ​​of the provided variables.
        :param variables: List or array of values ​​for the variables.
        :return: Array of output results.
        """
        self.aspen1.Reinit()
        self.aspen2.Reinit()
        # Before running Aspen evaluate if the point is feasible
        if self.Feasibily_Check(variables) == False:
            print("Unfeasible point tested, point is penalized")
            results =  self.Error_Run() 
            return results 
        Run_Status_Dir = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
        results = []  # List to save ressult
        if variables[2] == 0:  # Solvent one is tested (first simulation)
            if self.aspen1 is None:
                raise RuntimeError("ASPEN is not connected. Call connect() before running a simulation.")
            
            # Clean Aspen with default values
            self.Clean_Aspen()

            # Assigns the values ​​of the variables in ASPEN
            self.Update_Aspen(variables)

            # Run the simulation
            self.aspen1.Engine.Run2()

            if self.aspen1.Tree.FindNode(Run_Status_Dir) == None:
                results =  self.Error_Run()  
            else:
                Run_Status = self.aspen1.Tree.FindNode(Run_Status_Dir).Value
                if Run_Status == 9 or Run_Status == 10:
                    results =  self.Error_Run() 
                    self.aspen1.Reinit()
                else:
                    # Pureza
                    pureza = self.aspen1.Tree.FindNode(r"\Data\Streams\PRODUCT\Output\MOLEFRAC\MIXED\CAPRYC").Value
                    results.append(pureza)


                    # REBOILER  B1 #
                    # REB_DUTY_D1 = r"\Data\Blocks\D1\Output\REB_DUTY"
                    # ReboilerDuty = self.aspen1.Tree.FindNode(REB_DUTY_D1).Value*0.0041868    #cal/sec to kW
                    #results.append(-1*ReboilerDuty)
                    self.aspen1.Reinit()
        else:  # Other solvent
            if self.aspen2 is None:
                raise RuntimeError("ASPEN is not connected. Call connect() before running a simulation.")
            # Clean Aspen 
            self.Clean_Aspen()
            # Update Aspen
            self.Update_Aspen(variables)
            # Run simulation
            self.aspen2.Engine.Run2()

            if self.aspen2.Tree.FindNode(Run_Status_Dir) == None:
                results =  self.Error_Run()  
            else:
                Run_Status = self.aspen2.Tree.FindNode(Run_Status_Dir).Value
                if Run_Status == 9 or Run_Status == 10:
                    results =  self.Error_Run() 
                    self.aspen2.Reinit()
                else:
                    # PUREZa
                    pureza = self.aspen2.Tree.FindNode(r"\Data\Streams\PRODUCT\Output\MOLEFRAC\MIXED\CAPRYC").Value
                    # results.append(np.log(pureza))
                    # results.append(pureza)

                    # REBOILER  B1 #
                    REB_DUTY_D1 = r"\Data\Blocks\D1\Output\REB_DUTY"
                    ReboilerDuty = self.aspen2.Tree.FindNode(REB_DUTY_D1).Value*0.0041868    #cal/sec to kW
                    results.append(-1*ReboilerDuty)
                    self.aspen2.Reinit()

        return results

    def close(self):
        """Close the connection to ASPEN."""
        if self.aspen1 is not None:
            self.aspen1.Close()
            self.aspen1 = None
        if self.aspen2 is not None:
            self.aspen2.Close()
            self.aspen2 = None


# class AspenSimulation_MOBO:

class AspenSimulation_MOBO: # AspenSimulation:
    
    def __init__(self, aspen_file_path, output_nodes, listofcomponents):
        """
        Initializes the connection to ASPEN and sets the paths for variables and outputs.

        :param aspen_file_path: Path to the ASPEN file.
        :param variable_nodes: List of paths for variables in ASPEN.
        :param output_nodes: List of paths for outputs in ASPEN.
        """
        self.aspen_file_path = os.path.abspath(aspen_file_path)
        self.output_nodes = output_nodes
        self.listofcomponents = listofcomponents
        self.aspen = None

    def connect(self):
        import win32com.client as win32 
        """Connect to ASPEN and load the simulation"""
        self.aspen = win32.Dispatch("Apwn.Document") # .client.Dispatch("Apwn.Document")
        self.aspen.InitFromFile(self.aspen_file_path)
        self.aspen.Visible  = False
        self.aspen.SuppressDialogs = True
        time.sleep(2)
        self.aspen.Engine.Run2()
        time.sleep(2)

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
        x_in = x_ins[0]
        block_tag = tags[0]
        T  = Ts[0]
        FlashPoint = sum(FlashPoint_val*x_in)
        AutoIgnitionTemp = sum(IgnitionTemperature_val*x_in)
        Vol = (np.pi * ( d/2)**2)*(1.2*0.61*(nt-2))  # [m3]
        Pcolumn = float(self.aspen.Tree.FindNode(r"\Data\Blocks\{block}\Input\PRES1".format(block=block_tag)).Value*101.325)  # [atm a kPa]
        Mass =  float(self.aspen.Tree.FindNode(r"\Data\Blocks\{block}\Output\BAL_MASI_TFL".format(block=block_tag)).Value/3600)  # [Kg/sec]
        Treb = self.aspen.Tree.FindNode(r"\Data\Blocks\{block}\Output\BOTTOM_TEMP".format(block=block_tag)).Value + 273.15 # Celsius to Kelvyn
        #---------------F1,F2,F3,F4-----------------#
        Entalpy_comb = sum(abs(listHcom)*x_in )  # [KJ/mol] 
        
        F1 = 0.1*(Mass*(Entalpy_comb))/3.148
        F2 = (6/3.148)*Pcolumn*Vol
        Psat_in = self.Antoine_function(listA,listB, listC, listD, listE, listF, listG, T)*100 # [Bar a kPa]
        VapPress = sum(Psat_in*x_in )
        F3 = (1e-3)*(1/Treb)*((Pcolumn-VapPress)**2)*Vol   
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
        Fedi_val = 4.76*(Damage_Potential**(1/3)) 
        return Fedi_val

    def Feasibily_Check(self, x):
        Constraint_List = []
        Constraint_List.append(x[3]<x[4]) 
        Constraint_List.append(x[3]<x[5]) 
        Constraint_List.append(x[4]<x[5]) 
        Constraint_List.append(x[5]< (x[6]-3)) 
        Feasibility = all(Constraint_List) == True   
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
        self.aspen.Tree.FindNode(B2_INT_1).Value = x[5]   

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
        self.aspen.Tree.FindNode(B1_INT_1).Value = 60 - 1  


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
        return [0.001, -1000]  # composicion y FEDI

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

    def run_simulation(self, variables):
        """
        Runs the simulation in ASPEN with the values ​​of the provided variables.
        :param variables: List or array of values ​​for the variables.
        :return: Array of output results.
        """
        #----------------------Constraints function-------------#
        if self.aspen is None:
            raise RuntimeError("ASPEN is not connected. Call connect() before running a simulation..")
        
        # Before running Aspen, evaluate whether the point is feasible.
        if self.Feasibily_Check(variables) == False: 
            # print('Does not meet restrictions')
            results =  self.Error_Run() 
            return results 
        
         
        self.Clean_Aspen()
        self.Update_Aspen(variables)
        #-Run problem -#
        self.aspen.Engine.Run2()
            
        Run_Status_Dir = r"\Data\Results Summary\Run-Status\Output\UOSSTAT2"
        #First Check if Status return a value
        if self.aspen.Tree.FindNode(Run_Status_Dir) == None:  # This means there was a problem
            results = self.Error_Run()    
            return results
        else:
            Run_Status = self.aspen.Tree.FindNode(Run_Status_Dir).Value  
            B1Check= self.B1_Check(self.listofcomponents, variables[6])
            B2Check= self.B2_Check(self.listofcomponents, variables[5])
            if Run_Status == 9 or Run_Status == 10 or all(B1Check) == True or all(B2Check) == True: # Error in the simulation 
                results =  self.Error_Run() 
                self.aspen.Reinit()
                return results
            else: # Simulation without errors
                results = []
                nt1, nt2 = variables[6].item(), variables[5].item()       # Value of number of stages 
                d1 = self.aspen.Tree.FindNode(r"\Data\Blocks\B1\Subobjects\Column Internals\INT-1\Subobjects\Sections\INT\Input\CA_DIAM\INT-1\INT").Value
                d2 = self.aspen.Tree.FindNode(r"\Data\Blocks\B2\Subobjects\Column Internals\INT-1\Subobjects\Sections\INT\Input\CA_DIAM\INT-1\INT").Value
                        
                c1_mole_frac = r"\Data\Streams\DCM\Output\MOLEFRAC\MIXED\DICHL-01"
                c2_mole_frac = r"\Data\Streams\MEOH\Output\MOLEFRAC\MIXED\METHA-01"
                c3_mole_frac = r"\Data\Streams\DMF\Output\MOLEFRAC\MIXED\N:N-D-01"
                XD1  = self.aspen.Tree.FindNode(c1_mole_frac).Value
                results.append(XD1)
                listofcomponents = ['METHA-01','DICHL-01','N:N-D-01']
                FlashPoint_val = np.array([9 , -4 , 58]) +273.15                 # [K] 
                IgnitionTemperature_val = np.array([ 464, 605, 440]) + 273.15    # [K]
                NF, NR = np.array([3,1,2]),np.array([0,0,0])                     # NFPA values
                x_ins, Ts = self.variables_FEDI(listofcomponents,nt1=nt1)
                Fedi = self.Fedi(['B1'], listofcomponents, NF, NR, x_ins, Ts, max(d1,d2), nt1, FlashPoint_val, IgnitionTemperature_val)
                results.append(-Fedi)
                return results
    
    def close(self):
        """Closes the connection with ASPEN"""
        if self.aspen is not None:
            self.aspen.Close()
            self.aspen = None
