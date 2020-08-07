%==========================================================================
%This code simulates the MSE for the neural network when it attempts to
%predict a catalyst with a base metal it has not been trained upon.
%==========================================================================


clear
clc
close all



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

X1(:,[19:24]) = [];

x = zscore((X1(:,[1:size(X1,2)-1])))';

t = (zscore(X1(:,size(X1,2))))';

%% 

%Iterate Percentage of Data Trained on

p = [.8];

for k = 1:length(p)
    
%Create Indices for Unique Base Metals
    
[g,nan,uniqueind] = unique(X2(:,1));

for l = 1:length(g)

   
    
%Filter Unique Base Metals Each Iteration    

x2 = x(:,uniqueind==l);
t2 = t(:,uniqueind==l);

xc = x;
tc = t;

xc(:,uniqueind==l) = [];
tc(:,uniqueind==l) = [];

% Develop Test and Training Sets

Q = size(x, 1);
Q1 = floor(Q * p(k));
Q2 = Q - Q1;
ind = randperm(Q);
ind1 = ind(1:Q1);
ind2 = ind(Q1 + (1:Q2));

    
for j = 1:5



x1 = x(:, ind1);
t1 = t(:, ind1);
% x2 = x(:, ind2);
% t2 = t(:, ind2);

% Train Neural Networks

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
  y2 = neti(x2);
  perfs(i) = mse(neti, t2, y2);
  y2Total = y2Total + y2;
end
y2AverageOutput = y2Total / numNN;
perfAveragedOutputs2(k) = mse(t2, y2AverageOutput) 
plotregression(t2,y2AverageOutput)







%%

w3_1(l,j) = perfAveragedOutputs2(k);


end
end
end
%%

%Create Bar Plot for MSE of Missing Base Metal

for i = 1:7
    
    err(i) = sqrt(var(w3_1(i,:)'));
    
end

C = categorical({'Ir','Ru','Pt','Rh','Pd','Cu','Au'});

%%
close all
hold on
w = mean(w3_1(:,:)');
g = [w(7) w(6) w(1) w(5) w(3) w(4) w(2)];
err2 = [err(7) err(6) err(1) err(5) err(3) err(4) err(2)];
bar(C,w)
ylabel('Mean Square Error (MSE)','Interpreter','latex')
xlabel('Removed Primary Metal','Interpreter','latex')
errorbar(g,err2,'.')
set(gca,'fontsize',20)
grid on
box on
%%

