This folder contains the main code for the image-based MPC paper:

####### Model Training from 2D Heat Diffusion System #######
- DMD_Training.m: A Matlab script showing how to simulate a 2D heat diffusion system, and how to train the DMD low-order models based on the simulation data. The trained model should be saved as ‘**.mat’ file for further use by the MPC design in Julia functions.

- StandardMPC_Training.m: A Matlab file showing how to train a model for traditional MPC use. In traditional MPC, usually we have a set of thermostats located at specific locations for point wise measurement. This is different from image-based MPC where the measurement is taken by, e.g. thermal camera, which measures the entire field. The trained model should be saved as ‘**.mat’ file for further use by the MPC design in Julia functions.


######## DMD-MPC design ############
- Diff2d_Sim_6_0.jl: A Julia script, the main script for designing DMD-MPC. It contains the three reference field used in the paper for user to choose. It will call the other two scripts. The simulation results can be saved as .mat file using the last line. 

- steady_state_opt3.jl: A Julia script for the steady-state computation. The steady-state control may not be necessary for linear system. However, it will become necessary for the nonlinear MPC in the future. This script returns the steady-state input setpoint and state setpoint for dynamic MPC to follow. 

- LInearMPCSolverXsUs2.jl: A Julia for dynamic MPC to follow the setpoints provided by steady-state_opt3. It returns the optimal input sequence for the next iteration.

- Diff2D_Sim_6_0_standardMPC: Traditional MPC for controlling the 2D diffusion system. It contains the three reference field in the paper. Also, it consumes the model provided by ‘StandardMPC_Training.m’.

####### Data_ and_Plots ############
- This folder contains the Python script used to generate plots 5,8,12 in the paper. The data used by these Python scripts were saved during the running of previous Matlab and Julia code.

- The other plots in the paper are generated using OriginLab. Here we do not provide the corresponding data, since they are also saved from Matlab and Julia code, under different running scenarios. 

NOTE: Each time running these code may generate different result due to that the random seed was not fixed.