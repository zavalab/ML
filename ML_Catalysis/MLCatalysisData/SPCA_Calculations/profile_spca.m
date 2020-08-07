clear; close all; clc;

%% PROFILE
% Profile script for SPCA

n = 1000; p = 100; 
mu = zeros(1, p);
C = 0.5*ones(p) + (1-0.5)*eye(p);
X = mvnrnd(mu, C, n);
delta = 0.1;
K = 5;
stop = -10;

profile on
spca(X, [], K, delta, stop);
profile viewer

pause

n = 10; p = 1000; 
mu = zeros(1, p);
C = 0.5*ones(p) + (1-0.5)*eye(p);
X = mvnrnd(mu, C, n);
delta = 0.1;
K = 5;
stop = -100;

profile on
spca(X, [], K, delta, stop);
profile viewer
