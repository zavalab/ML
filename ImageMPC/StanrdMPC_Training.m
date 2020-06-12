% Author: Qiugang Lu 
% University of Wisconsin - Madison
% Reference: https://hplgit.github.io/fdm-book/doc/pub/diffu/pdf/diffu-4print.pdf
%
% Objectives:
%   Constructing the model used by standard MPC, where we have
%   thermostats at fixed set of locations: using DMD algorithm WITHOUT
%   dimension reduction

clear;clc
close all

Lx = 40; % Length
Ly = 40; % Width
Nx = 71; % Number of mesh grids
Ny = 71; % Number of mesh grids
dx = Lx/(Nx-1); % Spatial discretization step
dy = Ly/(Ny-1);
Lt = 4000; % Simulation duration in seconds  
Nt = 4001; % Number of snapshots
dt = Lt/(Nt-1); % Temporal discretization step

time = linspace(0,Lt,Nt);

alpha = 5.95;  % diffusion coefficient 
Fx = alpha*dt/dx^2;
Fy = alpha*dt/dy^2;

state = zeros(Ny,Nx);
u_n = zeros(Ny,Nx); % Current state snapshot

N = Ny*Nx; % Number of unknowns
A = sparse(N,N); % A matrix
b = zeros(N,1);  % Boundary condition

Statedata = zeros(6,6,length(time));
%% Fill out A
% index matrix m
for i = 1:Nx
   for j = 1:Ny
      m(i,j) = (j-1)*Nx + i; 
   end
end

j = 1;
for i = 1:Nx
   p = m(i,j);
   A(p,p) = 1;
end


for j = 2:Ny-1
   i = 1; p = m(i,j); A(p,p) = 1; 
   for i = 2:Nx-1
       p = m(i,j);
       A(p,m(i,j-1)) = -Fy;
       A(p,m(i-1,j)) = -Fx;
       A(p,p) = 1 + 2*Fx + 2*Fy;
       A(p,m(i+1,j)) = -Fx;
       A(p,m(i,j+1)) = -Fy;
   end
   i = Nx; p = m(i,j); A(p,p) = 1;
end


% equations corresponding to j = Ny, i = 1,2,3,...
j = Ny;
for i = 1:Nx
    p = m(i,j);
    A(p,p) = 1;
end


for n = 1:length(time)-1 
   % compute b 
   j = 1;
   for i = 1:Nx
       p = m(i,j); b(p) = 0;  % bottom boundary
   end
   
   for j = 2:Ny-1
       i = 1; p = m(i,j);
       b(p) = 0;  % left boundary
       
       i = Nx; p = m(i,j);
       b(p) = 0;  % right boundary
       
       for i = 2:Nx-1
           p = m(i,j);
           b(p) = u_n(Ny-j+1,i);
       end
       
   end
   j = Ny;
   for i = 1:Nx
       p = m(i,j); b(p) = 0; % upper boundary
   end

   
% %  Add source here

    f = zeros(Ny,Nx); % heat source input
    
  if mod(n,50) == 1
     [pindex,qindex] = meshgrid([16,24,32,38,46,55],[16,24,32,38,46,55]); % 6x6 heat sources
     f(sub2ind(size(f),pindex(:),qindex(:))) = randi(10,36,1)-5.6+0.1; % random input
     f_vec = (flip(f,1))';
     f_vec = f_vec(:); 
    
  end
    
    u_input(:,n) = f_vec(f_vec~=0); % store the heat source input values

    b = b + f_vec;
   
% solve the linear equation Ac = b
    c = A\b;
   
   % fill u with c
   for j = 1:Ny
       for i = 1:Nx          
           state(Ny-j+1,i) = c(m(i,j));
       end
   end
   
   
   % update the u_n
   u_n = state;
   
%%%%%%%%%%%%%%%%
   Statedata(:,:,n+1) = reshape(state(sub2ind(size(f),pindex(:),qindex(:))),6,6)';
%%%%%%%%%%%%%%%%

   if mod(n,100) == 1
       imagesc(state)
       %caxis([-5,15])
       colormap jet
       colorbar
       title(sprintf('Time = %.2f seconds',(n+1)*dt));
       pause(0.01)
%        xlim([11,60])
%        ylim([11,60])
   end
   
end

%% Data preparation
Udata_backup = Statedata;
Udata_backup = reshape(Udata_backup,[],Nt);
start = 1;
U = u_input(:,start:end);

data = reshape(Statedata,[],Nt);
dim = size(data,1);
mm = Nt - 1000; % Training data size is mm, test data size is 1000
X1 = data(:,1:mm-1);
Y1 = data(:,2:mm);
U1 = U(:,1:mm-1);

%% Solving least squares to obtain the model
Omega = [X1;U1];
[Uest,Sig,Vest] = svd(Omega,'econ');
thresh = 1e-10; % 
rtil = length(find(diag(Sig)>thresh*Sig(1,1)));

rtil = 36;

Util    = Uest(:,1:rtil); 
Sigtil  = Sig(1:rtil,1:rtil);
Vtil    = Vest(:,1:rtil); 

U_1 = Util(1:size(X1,1),:);
U_2 = Util(size(X1,1)+1:size(X1,1)+size(U1,1),:);

Aest = (Y1)*Vtil*inv(Sigtil)*U_1';
Best = (Y1)*Vtil*inv(Sigtil)*U_2';

%% save data
% save('Model_standard_MPC.mat','Aest','Best')