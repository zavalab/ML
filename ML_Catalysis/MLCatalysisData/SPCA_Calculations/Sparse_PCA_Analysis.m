clear
clc
close all


%============================================
%SPCA using LARS-EN Algorithm (SPASM Toolbox)
%============================================

load('data_table_all_points');

%Load Data to Array

A = table2array(data_table);

X= (table2array(data_table));

%fix Pt error

for i = 1:length(X)
    
    if X(i,1) == -5.1800
        
        X(i,1) = -5.8100;
        
    end
    
end

%Zscore or Normalize Data

X = zscore(X);

%Remove Binary Variables

X(:,(19:24)) = [];

var = 18;

%%
[n,p] = size(X(:,(1:var)));

% PCA and SPCA (Look to Spasm for Interpretation of these Variables)
[U, D, V] = svd(X(:,(1:var)), 'econ');
d = sqrt(diag(D).^2/n);
K = 2;
delta = inf;
stop = -7;
maxiter = 3000;
convergenceCriterion = 1e-9;
verbose = true;    

[B, SD, L] = spca(X(:,(1:var)), [], K, delta, stop, maxiter, convergenceCriterion, verbose);

%Create PCA and SPCA Projections

PCA = X(:,(1:var))*V(:,(1:2));
 
SPCA = X(:,(1:var))*B;

%Create Colors for Graphs

g = (A(:,12) == 172.1180 & A(:,1) == -5.1800 | A(:,12) == 172.1180 & A(:,1) == -3.0600);


for i = 1:length(A)
% 
    if A(i,1) == -3.0600 %gold
        Z(i) = 1;
        
    elseif A(i,1) == -3.7500
        Z(i) = 2;
    
    elseif A(i,1) == -5.8100
        Z(i) = 3;
        
    elseif A(i,1) == -5.6200
        Z(i) =4;
        
    elseif A(i,1) == -6.4600
        Z(i) =5;
        
    elseif A(i,1) == -5.6700
        Z(i) =6;
        
    elseif A(i,1) == -6.1600
        Z(i) = 7;
    
    else 
        
        Z(i) = 0;
        
    end
    
end


ya = SPCA(:,1);

na = SPCA(:,2);

colors = [0, 0.4470, 0.7410 ; 0.8500, 0.3250, 0.0980 ; 0.9290, 0.6940, 0.1250 ; 0.4940, 0.1840, 0.5560 ; 0.4660, 0.6740, 0.1880 ; 0.3010, 0.7450, 0.9330 ; 0.6350, 0.0780, 0.1840 ];

%Create SPCA Plots

figure(1)

gscatter(ya,na,X(:,1),colors,'o',10)
set(gca,'fontsize',18)
box on
grid on
legend({'Ir(Primary)','Ru(Primary)','Pt(Primary)','Rh(Primary)','Pd(Primary)','Cu(Primary)','Au(Primary)'},'Interpreter','latex','Location','northwest')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')

g = (A(:,3) ~= 0);

figure(3)
gscatter(ya,na,((-1*g) + 1),colors,'o',10)
set(gca,'fontsize',18)
box on
grid on
legend({'Catalyst With Promoter','Catalyst Without Promoter'},'Interpreter','latex','Location','northwest')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')


Z = (A(:,12) == 172.1180 & A(:,1) == -5.1800 & A(:,3) ~= 0 | A(:,12) == 172.1180 & A(:,1) == -3.0600 & A(:,3) ~= 0);

figure(2)
gscatter(ya,na,((-1*Z)+1),colors,'o',10)
set(gca,'fontsize',18)
box on
grid on
legend({'Au(CeO$$_2$$) and Pt(CeO$$_2$$) w/ promoter','Remaining Catalyst'},'Interpreter','latex','Location','northwest')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')
