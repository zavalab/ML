function SolvingLinearMPCXsUs(A,B,N,Q,QN,R,x0,umin,umax,xmin,xmax,xs,us,Uhat)
# benchmark example, works fine
# N is the prediction horizon
# QN, Q and R are weighting matrices

nx,nu = size(B)
mpc = Model(with_optimizer(Ipopt.Optimizer))

# define variables
@variable(mpc,x[i=1:nx,j=0:N])
@variable(mpc,umin[i] <= u[i=1:nu,j=0:N-1] <= umax[i])
@constraint(mpc, state[i=1:size(Uhat,1),k=0:N], -1.5*0 <= sum(Uhat[i,j]*x[j,k] for j=1:nx) <= 1.5) # only for Fig. 13 & 14 for
                                                                                                    # fast converge to steady state
# @constraint(mpc, state[i=1:size(Uhat,1),k=0:N], -1.0*0 <= sum(Uhat[i,j]*x[j,k] for j=1:nx) <= 1.0) # default

# define objective
@variable(mpc,obj)
@constraint(mpc,OBJ,obj==sum((x[i,N]-xs[i])*QN[i,j]*(x[j,N]-xs[j]) for i=1:nx,j=1:nx)
                            + sum((x[i,k]-xs[i])*Q[i,j]*(x[j,k]-xs[j]) for i=1:nx,j=1:nx,k=0:N-1)
                            + sum((u[i,k]-us[i])*R[i,j]*(u[j,k]-us[j]) for i=1:nu,j=1:nu,k=0:N-1))
@objective(mpc,Min,obj)

# define constraints
@constraint(mpc,Dynamic[i=1:nx,k=0:N-1],x[i,k+1] == sum(A[i,j]*x[j,k] for j=1:nx) + sum(B[i,j]*u[j,k] for j=1:nu))
@constraint(mpc,Initial[i=1:nx],x[i,0] == x0[i])

optimize!(mpc)

    return value.(x),value.(u)
end
