clear; close all; clc;

%% PROFILE
% Profile script for SLDA

%% create data set
p = 1000; % number of variables
nc = 500; % number of observations per class
n = 4*nc; % total number of observations
m1 = 0.6*[ones(10,1); zeros(p-10,1)]; % c1 mean
m2 = 0.6*[zeros(10,1); ones(10,1); zeros(p-20,1)]; % c2 mean
m3 = 0.6*[zeros(20,1); ones(10,1); zeros(p-30,1)]; % c3 mean
m4 = 0.6*[zeros(30,1); ones(10,1); zeros(p-40,1)]; % c4 mean
S = 0.6*ones(p) + 0.4*eye(p); % covariance is 0.6
c1 = mvnrnd(m1,S,nc); % class 1 data
c2 = mvnrnd(m2,S,nc); % class 2 data
c3 = mvnrnd(m3,S,nc); % class 3 data
c4 = mvnrnd(m4,S,nc); % class 3 data
X = [c1; c2; c3; c4]; % training data set
Y = [[ones(nc,1); zeros(3*nc,1)] [zeros(nc,1); ones(nc,1); zeros(2*nc,1)] [zeros(2*nc,1); ones(nc,1); zeros(nc,1)] [zeros(3*nc,1); ones(nc,1)]];

%% run algorithm
profile on
[beta theta] = slda(X,Y,1e-3,-100,1000);
profile viewer
