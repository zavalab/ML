clear all;
clc;

input_table = readtable('final_withoutPtAuCeO2_notNormalized.txt');


%% Neural Network Training

% Randomly assign data to training, validation and testing sets
X = table2array(input_table);
X = X' ;

R_best = 0.5;
MSE_best = 5;

for i=1:100 % Number of iterations
    i

    % Randomly assign data to training, validation and testing sets
    [dat_train,dat_test] = dividerand(X,0.8,0.2) ;

    r = length(dat_train(:,1));
        x_train = dat_train(1:r-1,:);
        y_train = dat_train(r,:);
    r = length(dat_test(:,1));
        x_test = dat_test(1:r-1,:);
        y_test = dat_test(r,:);


    % Create the Feedforward Neural Network
    size = [6 2];  
    net = fitnet(size, 'trainbr');
    net.trainParam.epochs = 2500;
    net.trainParam.showWindow = false; % display training GUI


    % Train the Neural Network
    net = train(net, x_train, y_train);
    ypred_train = net(x_train) ;
    R_train = (corr2(y_train,ypred_train))^2;
    MSE_train = immse(y_train,ypred_train);


    % Test the Neural Network
    ypred_test = net(x_test);
    R_test = (corr2(y_test,ypred_test))^2;
    MSE_test = immse(y_test,ypred_test);
    
    perf(i,1) = R_test;
    perf(i,2) = MSE_test;
    
    % Save the Neural Network
    if MSE_test < MSE_best
        MSE_best = MSE_test;
        % net_fullSet_20171203_WithoutAuPt = net;
        % save net_fullSet_20171203_WithoutAuPt
    end
end 
    

%% -------------------------------------------------------------------
% Report which data points were significantly different. 
disp("MSE Best") 
min(perf(:,2))

disp("R^2 Best") 
max(perf(:,1))











