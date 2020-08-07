%{
Given the file name for an ANN, it will re-plot the results. Given
additional data, it will use the ANN to make predictions and plot the
results of those predicitons. 
%}

clear all;
clc;

load net_fullSet_20171201_01
net = net_fullSet_20171201_01;

%% Fit with the Training Data

% Test the network, print out figure
y_train_predicted = net(x_train);
R2 = (corr2(y_train_predicted, y_train))^2;
MSE = immse(y_train_predicted, y_train);

figure('DefaultAxesFontSize',16)
scatter(y_train,y_train_predicted);
hold on 
plot([-10,0,10],[-10,0,10],'k');
xlabel('Experimental');
ylabel('Predicted');
xticks([-10 -5 0 5 10])
yticks([-10 -5 0 5 10])
string = compose("Training Performance\nMSE = " + round(MSE,3) + ", R^2 = " + round(R2,3));
title(string);


%% Fit with the non-training data

y_test_predicted = net(x_test);
MSE = immse(y_test_predicted, y_test);
R2 = (corr2(y_test_predicted, y_test))^2;

figure('DefaultAxesFontSize',16)
scatter(y_test_predicted, y_test);
hold on 
plot([-10,0,10],[-10,0,10],'k');
xlabel('Experimental');
ylabel('Predicted');
xticks([-10 -5 0 5 10])
yticks([-10 -5 0 5 10])
string = compose("Testing Performance\nMSE = " + round(MSE,3) + ", R^2 = " + round(R2,3));
title(string);


%% Fit with Au&Pt/CeO2

input_table = readtable('final_onlyPtAuCeO2.txt');

X = table2array(input_table);
X = X' ;

x_dat2 = X(1:23,:);
y_dat2 = X(24,:);

% Test the network, print out figure
y_pred2 = net(x_dat2); % 23 inputs
MSE = immse(y_dat2,y_pred2);

figure('DefaultAxesFontSize',18)
scatter(y_train,y_train_predicted,'b.');
hold on 
scatter(y_dat2,y_pred2,'r');
hold on
plot([-15,0,10],[-15,0,10],'k');
xlabel('Experimental');
ylabel('Predicted');
legend('Training Data','Au/CeO2 & Pt/CeO2 Instances');





