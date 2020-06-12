# https://hplgit.github.io/fdm-book/doc/pub/diffu/pdf/diffu-4print.pdf
using LinearAlgebra,PyPlot,SparseArrays,MAT,JuMP,Ipopt,Random#,GR
Random.seed!(1234)

Lx = 40; Ly = 40; Nx = 71; Ny = 71;
dx = Lx/(Nx-1); dy = Ly/(Ny-1);
Lt = 20000; Nt = 20000; # Simulation duration
dt = Lt/(Nt-1)

time = range(0,stop=Lt,length=Nt)

# alpha = 0.45
alpha = 5.95

Fx = alpha*dt/dx^2
Fy = alpha*dt/dy^2

state = zeros(Ny,Nx)

input = 4*(sin.(2*pi*0.02*0.1*(1:Nt).+3*pi/2).+1)

N = Ny*Nx # number of unknowns
# A = spzeros(N,N)
A = zeros(N,N)
b = zeros(N,1)

###############################################################################
## Fill out big A Matrix
# define the index function
m = zeros(Int32,Nx,Ny)
for i = 1:Nx
    for j = 1:Ny
         m[i,j] = (j-1)*Nx + i
    end
end

# define entries in A corresponds to j = 1, i = 1,2,3,...
j = 1
for i = 1:Nx
    p = m[i,j]
    A[p,p] = 1
end

for j = 2:Ny-1
    i = 1; p = m[i,j]; A[p,p] = 1
    for i = 2:Nx-1
        p = m[i,j]
        A[p,m[i,j-1]] = -Fx
        A[p,m[i-1,j]] = -Fy
        A[p,m[i,j]] = 1+2*Fx+2*Fy
        A[p,m[i+1,j]] = -Fx
        A[p,m[i,j+1]] = -Fy
    end
end

for j = Ny
    for i = 1:Ny
        p = m[i,j]
        A[p,p] = 1
    end
end
######################################
# Load parameters for controller

#### DMD model trained under different sample size #####
MATdataROM = matread("Model_under_Sample_4000.mat") # baseline
# MATdataROM = matread("Model_under_Sample_5000.mat") # DMD model under 5000 samples
# MATdataROM = matread("Model_under_Sample_3000.mat") # DMD model under 3000 samples
# MATdataROM = matread("Model_under_Sample_2000.mat") # DMD model under 2000 samples


Uhat = MATdataROM["Uhat"]

approxA = MATdataROM["approxA"]
approxB = MATdataROM["approxB"]

nr,nu = size(approxB)
Nx_patch = Ny_patch = sqrt(size(Uhat,1)) # dim of the selected region of interest

umin = -15*ones(nu,1)*0 #10
umax = 10*ones(nu,1)
xmin = -Inf*ones(nr,1) # in terms of xtilde in ROM, does not represent constraints on phyiscal states
xmax = Inf*ones(nr,1)


# objective weighting
Q = Diagonal(ones(nr));
Qf = Diagonal(2*ones(nr));
R = 0.1*Diagonal(ones(nu))

# # Reference signal
Nx_patch = Int(Nx_patch)
Ny_patch = Int(Ny_patch)
C = zeros(Nx_patch,Ny_patch)
############
# C = 0.8*ones(Nx_patch,Ny_patch)  ### Constatnt reference signal
############
# C = [exp(-0.001*((i-25)^2+(j-25)^2))  for i = 1:Nx_patch, j=1:Ny_patch] # Gaussuan shape reference
############
C = [exp(-0.001*((i-25)^2+(j-25)^2))  for i = 1:Nx_patch, j=1:Ny_patch] # Gaussuan shape reference
C[C.>0.8].=0.8
############

xref = C[:]

# project the reference on to the first coordinate as the actual Reference
xr = Uhat'*xref # ref in ROM space
horizon = 10

include("steady_state_opt3.jl")
include("LinearMPCSolverXsUs2.jl")

# Solve steady-state problem
xs,us = SteadyStateOptimization(approxA,approxB,xr,Uhat)

##############################################
# Iteratively obtain values
# NOTE: f is the source/input
Nsim = 30
Statedata = zeros(Ny,Nx,Nsim+1)
Statedata_innerPatch = zeros(length(11:60)^2, Nsim + 1)
Inputdata = zeros(nu,Nsim)

for n = 1:Nsim
    print("n=",n,"\n");
    # Compute b values at each time instant
    j = 1
    for i = 1:Nx
        p = m[i,j]; b[p] = 0;
    end

    for j = 2:Ny-1
        i = 1; p = m[i,j]; b[p] = 0;
        i = Nx; p = m[i,j]; b[p] = 0;
        for i = 2:Nx-1
            p = m[i,j]; b[p] = state[Ny-j+1,i]; # changed u_n to state
        end
    end

    j = Ny
    for i = 1:Nx
        p = m[i,j]; b[p] = 0;
    end


## Add sources
    global f = zeros(Ny,Nx) # local variable

    ############################# NOTE SOLVE MPC based on ROM ##################
    state_innerpatch = state[11:60,11:60]
    x0_ROM = Uhat'*state_innerpatch[:]
    global xopt,fopt = SolvingLinearMPCXsUs(approxA,approxB,horizon,Q,Qf,R,x0_ROM,umin,umax,xmin,xmax,xs,us,Uhat)

# 6 x 6
    f[16,16] = fopt[31,0]; f[16,24] = fopt[32,0]; f[16,32] = fopt[33,0]; f[16,38] = fopt[34,0]; f[16,46] = fopt[35,0]; f[16,55] = fopt[36,0];
    f[24,16] = fopt[25,0]; f[24,24] = fopt[26,0]; f[24,32] = fopt[27,0]; f[24,38] = fopt[28,0]; f[24,46] = fopt[29,0]; f[24,55] = fopt[30,0];
    f[32,16] = fopt[19,0]; f[32,24] = fopt[20,0]; f[32,32] = fopt[21,0]; f[32,38] = fopt[22,0]; f[32,46] = fopt[23,0]; f[32,55] = fopt[24,0];
    f[38,16] = fopt[13,0]; f[38,24] = fopt[14,0]; f[38,32] = fopt[15,0]; f[38,38] = fopt[16,0]; f[38,46] = fopt[17,0]; f[38,55] = fopt[18,0];
    f[46,16] = fopt[7,0]; f[46,24] = fopt[8,0]; f[46,32] = fopt[9,0]; f[46,38] = fopt[10,0]; f[46,46] = fopt[11,0]; f[46,55] = fopt[12,0];
    f[55,16] = fopt[1,0]; f[55,24] = fopt[2,0]; f[55,32] = fopt[3,0]; f[55,38] = fopt[4,0]; f[55,46] = fopt[5,0]; f[55,55] = fopt[6,0];

    global f_vec = f[end:-1:1,:]
    f_vec = convert(Array{Float64},reshape(f_vec',:,1))
    Inputdata[:,n] = fopt[:,0]
    ############################################################################

    global b = b + f_vec

    c = (A+1e-9*sparse(I,size(A,1),size(A,2)))\b # add small positive to avoid inverting a singular A

    # fill the state with c
    for j = 1:Ny
        for i = 1:Nx
            state[Ny-j+1,i] = c[m[i,j]]
        end
    end

    # update the Statedata
    Statedata[:,:,n+1] = state
    Statedata_innerPatch[:,n+1] = state[11:60,11:60][:]

    # plot the simulation results
    clf()
    figure(1)
    surf(state[11:60,11:60],cmap="jet")
end

# matwrite("julia_rk25.mat", Dict("input_julia_rk25" => Inputdata, "state_julia_rk25" => Statedata_innerPatch, "ref_rk25" => C); compress=true)
