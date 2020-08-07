function [X mu] = center(X)
%CENTER Center the columns (variables) of a data matrix to zero mean.
%
%   X = CENTER(X) centers the observations of a data matrix such that each
%   variable (column) has zero mean.
%
%   [X MU] = CENTER(X) also returns a vector MU of mean values for each
%   variable. 
%
%   This function is an auxiliary part of SpaSM, a matlab toolbox for
%   sparse modeling and analysis.
%
%  See also NORMALIZE.

n = size(X,1);
mu = mean(X);
X = X - ones(n,1)*mu;
