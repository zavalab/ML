# https://hplgit.github.io/fdm-book/doc/pub/diffu/pdf/diffu-4print.pdf
using LinearAlgebra,PyPlot,SparseArrays,MAT,JuMP,Ipopt,Random#,GR
Random.seed!(1234)

# NOTE: NOTE NOTE NOTE: For Fig 13 & 14, for fast convergence to steady state,
# please set the state constraint to [0,1.5]


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

MATdataROM = matread("Model_standard_MPC.mat") #traditional MPC

# Uhat is set as an identifty matrix, since standard MPC does not need projecting to low-dim space
Uhat = Matrix(1I,50,50)

approxA = MATdataROM["Aest"]
approxB = MATdataROM["Best"]

nr,nu = size(approxB)
Nx_patch = Ny_patch = 50 # dim of the selected region of interest

umin = -15*ones(nu,1)*0 #0
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

xref_matrix =  [C[6,6] C[6,14] C[6,22] C[6,28] C[6,36] C[6,45];
                C[14,6] C[14,14] C[14,22] C[14,28] C[14,36] C[14,45];
                C[22,6] C[22,14] C[22,22] C[22,28] C[22,36] C[22,45];
                C[28,6] C[28,14] C[28,22] C[28,28] C[28,36] C[28,45];
                C[36,6] C[36,14] C[36,22] C[36,28] C[36,36] C[36,45];
                C[45,6] C[45,14] C[45,22] C[45,28] C[45,36] C[45,45]];
xref = xref_matrix[:]

xr = xref
horizon = 10
include("steady_state_opt3.jl")
include("LinearMPCSolverXsUs2.jl")

# Solve steady-state problem
xs,us = SteadyStateOptimization(approxA,approxB,xr,Uhat)


##############################################
# Iteratively obtain values
# NOTE: f is the source/input
Nsim = 100#30
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
    Z = state_innerpatch
    x0_ROM = [Z[6,6]  Z[6,14]  Z[6,22]  Z[6,28]  Z[6,36]  Z[6,45];
              Z[14,6] Z[14,14] Z[14,22] Z[14,28] Z[14,36] Z[14,45];
              Z[22,6] Z[22,14] Z[22,22] Z[22,28] Z[22,36] Z[22,45];
              Z[28,6] Z[28,14] Z[28,22] Z[28,28] Z[28,36] Z[28,45];
              Z[36,6] Z[36,14] Z[36,22] Z[36,28] Z[36,36] Z[36,45];
              Z[45,6] Z[45,14] Z[45,22] Z[45,28] Z[45,36] Z[45,45]];

    global xopt,fopt = SolvingLinearMPCXsUs(approxA,approxB,horizon,Q,Qf,R,x0_ROM[:],umin,umax,xmin,xmax,xs,us,Uhat)

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

    # c = A\b  # A is singular not doable
    c = (A+1e-9*sparse(I,size(A,1),size(A,2)))\b

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
