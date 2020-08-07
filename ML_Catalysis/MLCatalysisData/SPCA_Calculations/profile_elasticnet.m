clear; close all; clc;

%% PROFILE
% Profile script for Elastic Net

n = 100000; p = 100; 
mu = zeros(1, p);
C = 0.5*ones(p) + (1 -0.5)*eye(p);
X = mvnrnd(mu, C, n);
y = center(rand(n,1));
delta = 0.1;

profile on
elasticnet(X, y, delta);
profile viewer

pause

n = 100; p = 10000; 
mu = zeros(1, p);
C = 0.5*ones(p) + (1 -0.5)*eye(p);
X = mvnrnd(mu, C, n);
y = center(rand(n,1));
delta = 0.1;

profile on
elasticnet(X, y, delta);
profile viewer
