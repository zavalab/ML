%===================================================================
%This code Allows for Recreation of the PCA Analysis from the Paper
%===================================================================
clear
clc

%Load Preformed Data Table

load('data_table_all_points.mat')

A = table2array(data_table);

%Remove Reaction Rate 

A(:,35) = [];

%Remove Binary Variables (Preparation Methods)

A(:,19:25) = [];



for i = 1:length(A)
    
    if A(i,1) == -5.1800
        
        A(i,1) = -5.8100;
        
    end
end

%Zscore Data for PCA Analysis

[A1,mu,sigma] = zscore(A);

%Calculate Correlation Coefficients of A

C = corrcoef(A);

%Find Eigenvalues of Correlation Matrix

[V,D] = eig(C);

%Sort the Eigenvalues from largest to smallest

[d,ind] = sort(diag(D),'descend');

Ds = D(ind,ind);

Vs = V(:,ind);

%Calculate Principal Component Directions by projection data on to
%Eigenvectors with top 3 largest Eigenvalues.

PC1 = Vs(:,1)'*A1';

PC2 = Vs(:,2)'*A1';

PC3 = Vs(:,3)'*A1';

%Set Color Scheme for Graphs
colors = [0, 0.4470, 0.7410 ; 0.8500, 0.3250, 0.0980 ; 0.9290, 0.6940, 0.1250 ; 0.4940, 0.1840, 0.5560 ; 0.4660, 0.6740, 0.1880 ; 0.3010, 0.7450, 0.9330 ; 0.6350, 0.0780, 0.1840 ];

%Plot PCA Graph with Colored Primary Metals

close all
figure(1)
gscatter(PC1,PC2,A(:,1),colors,'o',8)
set(gca,'fontsize',26)
box on
grid on
legend({'Ir(Primary)','Ru(Primary)','Pt(Primary)','Rh(Primary)','Pd(Primary)','Cu(Primary)','Au(Primary)'},'Interpreter','latex','Location','northwest')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')


%Plot PCA Graph with Promoter Vs. No Promoter Colored

g = (A(:,3) ~= 0);

figure(2)
gscatter(PC1,PC2,g,colors,'o',8)
set(gca,'fontsize',25)
box on
grid on
legend({'Catalyst Without Promoter','Catalyst With Promoter'},'Interpreter','latex','Location','northwest')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')


%Plot PCA Graph with Experimental Temperature

figure(3)
scatter(PC1,PC2,[],A(:,21))
set(gca,'fontsize',25)
box on
grid on
h = colorbar;
ylabel(h, 'Temperature ($K^o$)','Interpreter','latex')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')

%%

%Create PCA Graphs using only Catalyst Information, and not including
%Experimental Conditions

%Calculate Correlation Coefficients of A

C = corrcoef(A(:,[1:18]));

%Find Eigenvalues of Correlation Matrix

[V,D] = eig(C);

%Sort the Eigenvalues from largest to smallest

[d,ind] = sort(diag(D),'descend');

Ds = D(ind,ind);

Vs = V(:,ind);

%Calculate Principal Component Directions by projection data on to
%Eigenvectors with top 3 largest Eigenvalues.

PC1 = Vs(:,1)'*A1(:,[1:18])';

PC2 = Vs(:,2)'*A1(:,[1:18])';

PC3 = Vs(:,3)'*A1(:,[1:18])';


%Set Color Scheme for Graphs
colors = [0, 0.4470, 0.7410 ; 0.8500, 0.3250, 0.0980 ; 0.9290, 0.6940, 0.1250 ; 0.4940, 0.1840, 0.5560 ; 0.4660, 0.6740, 0.1880 ; 0.3010, 0.7450, 0.9330 ; 0.6350, 0.0780, 0.1840 ];

%Plot PCA Graph with Colored Primary Metals

figure(4)
gscatter(PC1,PC2,A(:,1),colors,'o',8)
set(gca,'fontsize',26)
box on
grid on
legend({'Ir(Primary)','Ru(Primary)','Pt(Primary)','Rh(Primary)','Pd(Primary)','Cu(Primary)','Au(Primary)'},'Interpreter','latex','Location','northwest')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')




%Plot PCA Graph with Promoter Vs. No Promoter Colored

g = (A(:,3) ~= 0);

figure(5)
gscatter(PC1,PC2,g,colors,'o',8)
set(gca,'fontsize',25)
box on
grid on
legend({'Catalyst Without Promoter','Catalyst With Promoter'},'Interpreter','latex','Location','northwest')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')







