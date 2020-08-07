%==========================================================================
%This code provides the method for understanding the impact of percent
%trained data upon the accuracy of the network.
%==========================================================================


clear
clc
close all

rng(1)

%Load Raw Experimental Data
load('data_table_all_points');
load('Catalyst_Data.mat')
load('Promoter_Data.mat')
load('Support_Data.mat')
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

%% Create a Dataset of Pt on CeO2 with every promoter in the dataset, can we find one that might be better?

for i = 1:28
    Xtest(i,:) = [Catalyst(1),1,Promoter(i,:),10,Support(3,:),X1(162,19:28)];
end  

x1 = x;
t1 = t;
%% Train Neural Network on Data
  

%Set Parameters for Neural Networks
net = feedforwardnet([6 2],'trainbfg');
net.divideFcn = '';
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;
net.performParam.regularization = 0.5;


%Select Number of Neural NEtworks in Ensemble
numNN = 5;
nets = cell(1, numNN);
for i = 1:numNN
  fprintf('Training %d/%d\n', i, numNN)
  nets{i} = train(net, x1, t1);
end


%%

% Evaluate Performance of Networks

perfs = zeros(1, numNN);
y2Total = 0;
for i = 1:numNN
  neti = nets{i};
  y2 = neti(Xtest');
  y2Total = y2Total + y2;
end
y2AverageOutput = y2Total / numNN;



%%
% close all
% hold on
% 
% plot(p,w,'LineWidth',2)
% ylabel('Mean Square Error (MSE)','Interpreter','latex')
% xlabel('Percent Trained Data','Interpreter','latex')
% errorbar(p,w,err,'.')
% set(gca,'fontsize',20)
% grid on
% box on
%%

