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


class AspenSimulation_MOBO:
    
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
        return [0.0001, -1000] #

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
                    # PUREZa
                    pureza = self.aspen1.Tree.FindNode(r"\Data\Streams\PRODUCT\Output\MOLEFRAC\MIXED\CAPRYC").Value
                    results.append(pureza)


                    # REBOILER  B1 #
                    REB_DUTY_D1 = r"\Data\Blocks\D1\Output\REB_DUTY"
                    ReboilerDuty = self.aspen1.Tree.FindNode(REB_DUTY_D1).Value*0.0041868    #cal/sec to kW
                    results.append(-1*ReboilerDuty)
                    self.aspen1.Reinit()
        else: # Other solvent
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
                    results.append(pureza)

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





