%==========================================================================
%This code allows you to recreate the NN regression plots, and also allows
%for the exclusion of features for the regression.
%==========================================================================

clear
clc
close all

rng(1)

%Load Raw Experimental Data
load('data_table_all_points');

X1 = (table2array(data_table));

%fix Pt error

for i = 1:length(X1)
    
    if X1(i,1) == -5.1800
        
        X1(i,1) = -5.8100;
        
    end
    
end

%Separate Data into Inputs (x) and Outputs (t)
X2 = X1;

%Remove Binary Variables
X1(:,[19:24]) = [];

x = zscore((X1(:,[1:size(X1,2)-1])))';

t = (zscore(X1(:,size(X1,2))))';

%% 

%Iterate Percentage of Data Trained on

p = [0.3];

    
%Exclude Particular Inputs    

%Exclude Table 7
%x(:,[32 33 4 9 5 8 6 7 10 3 11 15 14 18 16 17 12 13]) = [];    




    
% Develop Test and Training Sets

Q = size(x, 2);
Q1 = floor(Q * p);
Q2 = Q - Q1;
ind = randperm(Q);
ind1 = ind(1:Q1);
ind2 = ind(Q1 + (1:Q2));    

x1 = x(:, ind1);
t1 = t(:, ind1);
x2 = x(:, ind2);
t2 = t(:, ind2);

% Train Neural Networks

%Set Parameters for Neural Networks
net = feedforwardnet([6 2],'trainbfg');
net.divideFcn = '';
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;
net.performParam.regularization = 0.5;


%Select Number of Neural Networks in Ensemble and train them
numNN = 5;
nets = cell(1, numNN);
for i = 1:numNN
  fprintf('Training %d/%d\n', i, numNN)
  nets{i} = train(net, x1, t1);
end


%%

% Evaluate Performance of Networks (MSE)

perfs = zeros(1, numNN);
y2Total = 0;
for i = 1:numNN
  neti = nets{i};
  y2 = neti(x2);
  perfs(i) = mse(neti, t2, y2);
  y2Total = y2Total + y2;
end
y2AverageOutput = y2Total / numNN;
perfAveragedOutputs2 = mse(t2, y2AverageOutput) 

%Set Color Scheme for Graphs
colors = [0, 0.4470, 0.7410 ; 0.8500, 0.3250, 0.0980 ; 0.9290, 0.6940, 0.1250 ; 0.4940, 0.1840, 0.5560 ; 0.4660, 0.6740, 0.1880 ; 0.3010, 0.7450, 0.9330 ; 0.6350, 0.0780, 0.1840 ];


close all
gscatter(t2,y2AverageOutput,X1(ind2,:),colors,'o',8)
set(gca,'fontsize',20)
colormap winter
box on
grid on
xlim([-2 2])
ylim([-4 2])
refline(1,0)
legend({'Ir(Primary)','Ru(Primary)','Pt(Primary)','Rh(Primary)','Pd(Primary)','Cu(Primary)','Au(Primary)'},'Interpreter','latex','Location','northwest')
xlabel('Actual log(Reaction Rate) ','Interpreter','latex')
ylabel('Predicted  log(Reaction Rate) ','Interpreter','latex')





