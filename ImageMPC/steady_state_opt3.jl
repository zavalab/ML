function SteadyStateOptimization(A,B,xr,Uhat)
# steady-state optimization for target tracking

## construct the steady-state optimization problem
nx,nu = size(B)
steadystate = Model(with_optimizer(Ipopt.Optimizer))

n = size(Uhat,1)
# define variables
@variable(steadystate,xs[i=1:nx])
    @variable(steadystate, -15.0*0 <= us[i=1:nu] <= 10.0)

# define objective
@objective(steadystate,Min,sum((xs[i]-xr[i])^2/nx for i = 1:nx if i!=nx))
#
# define constraints: xs = A*xs + B*us
@constraint(steadystate,System[i=1:nx],xs[i] == sum(A[i,j]*xs[j] for j=1:nx) + sum(B[i,j]*us[j] for j = 1:nu))

# add constraint on 2500-d state profile
# @constraint(steadystate,State[i=1:n], -1.0*0 <= sum(Uhat[i,j]*xs[j] for j=1:nx) <= 1.0) # default
@constraint(steadystate,State[i=1:n], -1.0*0 <= sum(Uhat[i,j]*xs[j] for j=1:nx) <= 1.5) ### For fig. 13 & 14 only, to converge faster

# Solving Optimization
optimize!(steadystate)

    return value.(xs), value.(us)
end
