%==========================================================================
%This code provides the optimization algorithm for identifying new catalyst
%formulations in the PCA space.

%Note: Some catalyst formulations may appear far away within the pca space,
%in order to correct this, a re-initialization of the algorithm with a new
%random seed may be required.
%==========================================================================


clear
clc
close all

%Load All Possible Catalyst, Promoter, and Support Combinations

load 'Catalyst_Data'

Catalyst1 = Catalyst;

load 'Promoter_Data'

Promoter1 = Promoter;

load 'Support_Data'

Support1 = Support;

load 'data_table_all_points.mat'

rng(1)

%data_table.Reference = [];

A = table2array(data_table);

for i = 1:length(A)
    
    if A(i,1) == -5.1800
        
        A(i,1) = -5.8100;
        
    end
end

[A1,mu,sigma] = zscore(A);

C = corrcoef(A(:,[1:18]));

[V,D] = eig(C);

[d,ind] = sort(diag(D),'descend');

Ds = D(ind,ind);

Vs = V(:,ind);

PC1 = Vs(:,1)'*A1(:,[1:18])';

PC2 = Vs(:,2)'*A1(:,[1:18])';

hold on
scatter(PC1,PC2,50)


%Normalize Data Points

Catalyst = (Catalyst - mu(1))/sigma(1);

Promoter = (Promoter - repelem(mu(3:10),28,1))./sigma(3:10);

Support = (Support - repelem(mu(12:18),18,1))./sigma(12:18);


%===========Target Values==============% 

Phat1 = .75;
Phat2 = -4;

%=========== Optimization Problem (Normal PCA) ==============%

%Initialize Xo
x0 = rand(1,18)';


%Function Definition

fun  = @(x) (Vs(:,1)'*x - Phat1)^2 + (Vs(:,2)'*x - Phat2)^2;

%Constraints for Variable Vector

%Place -inf or inf (physical requirements) on system

lb1 = min(A(:,1:18));

%Create Softmax filter (Lower Bound)

for i = 1:length(lb1)
    
    if lb1(i) > 0 
        
        lb1(i) = 0;
        
    else 
        
        lb1(i) = 2*lb1(i);
        
    end
    
    
end


lb = (lb1 - mu(1:18))./sigma(1:18);

%Upper Bound

ub1 = max(A(:,1:18));

%Create Softmax filter (Upper Bound)

for i = 1:length(ub1)
    
    if ub1(i) < 0 
        
        ub1(i) = 0;
        
    else 
        
        ub1(i) = 2*ub1(i);
        
    end
    
end

%Upper Bound on Percent Loading

ub = (ub1 - mu(1:18))./sigma(1:18);


% Calculation of Variable (x)

x = fmincon(fun,x0,[],[],[],[],lb,ub);


val = (x'.*sigma(1:18))' + mu(1:18)';


%Create Reference Points

xc = x(1);

xp = x(3:10);

xs = x(12:18);

%Minimize distance between optimal and true physical catalsyt

%Catalyst Matching

for i = 1:length(Catalyst)
    
    CatDist(i) = sqrt((Vs(1,1)*Catalyst(i) - Vs(1,1)*xc)^2 + (Vs(1,2)*Catalyst(i)  - Vs(1,2)*xc)^2);
    
end

[m1,cat_select] = min(CatDist);


for i = 1:size(Promoter,1)
    
    PromDist(i) = norm((Vs(3:10,1)'*Promoter(i,:)' - Vs(3:10,1)'*xp)^2 + (Vs(3:10,2)'*Promoter(i,:)' - Vs(3:10,2)'*xp)^2);
    
end

[m2,prom_select] = min(PromDist);

for i = 1:size(Support,1)
    
    SuppDist(i) = norm((Vs(12:18,1)'*Support(i,:)' - Vs(12:18,1)'*xs)^2 + (Vs(12:18,2)'*Support(i,:)' - Vs(12:18,2)'*xs)^2);
    
end

[m3,supp_select] = min(SuppDist);

xnew = [Catalyst(cat_select),x(2),Promoter(prom_select,:),x(11),Support(supp_select,:)];

%Distance between target point and actual catalyst

Dist1 = sqrt((Vs(:,1)'*xnew' - Vs(:,1)'*x)^2 + (Vs(:,2)'*xnew' - Vs(:,2)'*x)^2);

%Check to see if Removal of Promoter is Necessary

if Dist1 >= 1
    
    Promoter1(29,:) = zeros(1,8);
    
    Promoter(29,:) = (Promoter1(29,:) - mu(3:10))./sigma(3:10);
    
    xnone = 0;
    
    prom_none = 29;
    
    xnew1 = [Catalyst(cat_select),x(2),Promoter(prom_none,:),xnone,Support(supp_select,:)];

    Dist2 = sqrt((Vs(:,1)'*xnew1' - Vs(:,1)'*x)^2 + (Vs(:,2)'*xnew1' - Vs(:,2)'*x)^2);

    if Dist1 > Dist2
        
        %Resolve Optimization without Promoter
        
        %Index of Non Promoter Variables
        
        index = [3:11];
        
        %Optimization 
        
        fun  = @(x) (Vs(:,1)'*x - Phat1)^2 + (Vs(:,2)'*x - Phat2)^2;
        
        lb1(index) = 0;
        
        ub1(index) = 0;
        
        lb = (lb1 - mu(1:18))./sigma(1:18);
        
        ub = (ub1 - mu(1:18))./sigma(1:18);
        
        x = fmincon(fun,x0,[],[],[],[],lb,ub);
        
        val = (x'.*sigma(1:18))' + mu(1:18)';
        
        for i = 1:length(Catalyst)
    
            CatDist(i) = sqrt((Vs(1,1)*Catalyst(i) - Vs(1,1)*xc)^2 + (Vs(1,2)*Catalyst(i)  - Vs(1,2)*xc)^2);
    
        end

        [m1,cat_select] = min(CatDist);
        
        for i = 1:size(Support,1)
    
            SuppDist(i) = sqrt((Vs(12:18,1)'*Support(i,:)' - Vs(12:18,1)'*xs)^2 + (Vs(12:18,2)'*Support(i,:)' - Vs(12:18,2)'*xs)^2);
    
        end

        [m3,supp_select] = min(SuppDist);
        
        prom_select = 29;
        
        xnew = [Catalyst(cat_select),x(2),Promoter(prom_select,:),xnone,Support(supp_select,:)];
    
        
    end
    
    Promoter(29,:) = [];
    
end



%Create New Catalyst Selection Point
g = Catalyst1(cat_select);
X = sprintf(' Catalyst Binding Energy is (C) %d',g);
disp(X)
h = Promoter1(prom_select,1);

H = sprintf('Promoter MW is %d',h);
disp(H)

j = Support1(supp_select,2);

J = sprintf('Support MW is %d',j);

disp(J)

%Plot New Catalysts on PCA projections

scatter(Vs(:,1)'*xnew',Vs(:,2)'*xnew',150,'Fill')

scatter(Vs(:,1)'*x,Vs(:,2)'*x,150,'*')

set(gca,'fontsize',26)
box on
grid on
legend({'Existing Catalysts','Optimized Catalyst','Target Point'},'Interpreter','latex')
xlabel('Principal Component 1','Interpreter','latex')
ylabel('Principal Component 2','Interpreter','latex')