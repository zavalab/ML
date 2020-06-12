% Author: Qiugang Lu 
% University of Wisconsin - Madison
% Reference: https://hplgit.github.io/fdm-book/doc/pub/diffu/pdf/diffu-4print.pdf
%
% Objectives:
%   1. Constructing the 2D heat diffusion system, use forward finite difference approach
%   2. Training the DMDc algorithm

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
       p = m(i,j); b(p) = 0; % bottom boundary
   end
   
   for j = 2:Ny-1
       i = 1; p = m(i,j);
       b(p) = 0; % left boundary
       
       i = Nx; p = m(i,j);
       b(p) = 0; % right boundary
       
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
   
   % save the data
   Statedata(:,:,n+1) = state;%u(51:150,51:150);
%    
   if mod(n,100) == 1
       imagesc(state)
       %caxis([-5,15])
       colormap jet
       colorbar
       title(sprintf('Time = %.2f seconds',(n+1)*dt));
       pause(0.01)
   end
   
end

%% Data preparation
Udata_backup = Statedata;
Udata_backup = reshape(Udata_backup,[],Nt);
start = 1;
Statedata = Statedata(11:60,11:60,start:end); % Only use the state in the inner region
U = u_input(:,start:end);

data = reshape(Statedata,[],Nt);
dim = size(data,1);
mm = Nt - 1000; % Training data size is mm, test data size is 1000
X1 = data(:,1:mm-1);
Y1 = data(:,2:mm);
U1 = U(:,1:mm-1);

%% DMD estimate
thresh = 1e-3;
[Phi,lambda,approxA,approxB,r,Uhat,Phi_leftInverse,Aest,Best] = DMDHeat_2D(X1,Y1,U1,thresh);

% The test data
Xdata_m = data(:,mm+1:end);
U_m = u_input(:,mm+1:end);

% predicting the future
xpred = zeros(r,size(Xdata_m,2));
xpred(:,1) = Uhat'*Xdata_m(:,1);

for t = 1:size(Xdata_m,2)-1
    xpred(:,t+1) = approxA*xpred(:,t) + approxB*U_m(:,t);
end
xpred_rec = Uhat*xpred;

xpred_rec_reshape = reshape(xpred_rec,50,50,[]); % predicted state field
Xdata_m_reshape = reshape(Xdata_m,50,50,[]); % true state field


h4 = figure(209);
set(h4, 'Position',[61,144,1592,669],'PaperSize',[20 10]);

kk = 0;
for k = 200:200:size(Xdata_m,2)
   
   kk = kk + 1;
   num_plots = length(200:200:size(Xdata_m,2));
   subplot(2,num_plots,kk)
   imagesc((Xdata_m_reshape(:,:,k)))
   truemap(:,:,kk) = Xdata_m_reshape(:,:,k);
   colormap jet
%    caxis([-0.31,0.31])
%    zlim([0,1])
%    colorbar
   xlabel(sprintf('$t$ = %.0f s', k + mm - 1),'fontsize',20,'interpreter','latex')
   if kk == 1
      ylabel('State field (true)','fontsize',20,'interpreter','latex') 
   end
   
   
   subplot(2,num_plots,kk+num_plots)
   imagesc((xpred_rec_reshape(:,:,k)))
   predmap(:,:,kk) = xpred_rec_reshape(:,:,k);
   colormap jet
%    caxis([-0.31,0.31])
%    zlim([0,1])
%    colorbar
   xlabel(sprintf('$t$ = %.0f s', k + mm - 1),'fontsize',20,'interpreter','latex')
   if kk == 1
      ylabel('State field (DMDc)','fontsize',20,'interpreter','latex') 
   end
   
   Umap(:,:,kk) = (flipud(reshape(U_m(:,k-1),6,6)'));
   pause(0.5)
end
h = colorbar;
set(h, 'Position', [0.9314 .10 .0081 .8250]);


%% save the data
% save('data.mat','Aest','approxA','approxB','Best','Uhat')
%%
function [Phi,lambda,approxA,approxB,r,Uhat,Phi_leftInverse,Aest,Best] = DMDHeat_2D(X,Xp,U,epsilon)

Ups = U;
Omega = [X;Ups];
[U,Sig,V] = svd(Omega,'econ');

thresh = epsilon;
rtil = length(find(diag(Sig)>thresh*Sig(1,1)));

rtil = 50 %baseline is 50

Util    = U(:,1:rtil); 
Sigtil  = Sig(1:rtil,1:rtil);
Vtil    = V(:,1:rtil); 

[U,Sig,V] = svd(Xp);

thresh = epsilon;
r = length(find(diag(Sig)>thresh*Sig(1,1)));
r = 40 % baseline is 40

Uhat    = U(:,1:r); 
Sighat  = Sig(1:r,1:r);
Vbar    = V(:,1:r); 

n = size(X,1); 
q = size(Ups,1);
U_1 = Util(1:n,:);
U_2 = Util(n+1:n+q,:);

approxA = Uhat'*(Xp)*Vtil*inv(Sigtil)*U_1'*Uhat;
approxB = Uhat'*(Xp)*Vtil*inv(Sigtil)*U_2';

Aest = (Xp)*Vtil*inv(Sigtil)*U_1';
Best = (Xp)*Vtil*inv(Sigtil)*U_2';

[W,D] = eig(approxA);

lambda = diag(D); 

Phi = Uhat * W;
Phi_leftInverse = W\Uhat';

end