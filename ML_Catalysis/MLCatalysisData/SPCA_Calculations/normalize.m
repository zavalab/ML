function [X mu d] = normalize(X)
%NORMALIZE Normalize the columns (variables) of a data matrix to unit
%Euclidean length.
%
%   X = NORMALIZE(X) centers and scales the observations of a data matrix
%   such that each variable (column) has unit Euclidean length. For a
%   normalized matrix X, X'*X is equivalent to the correlation matrix of X.
%
%   [X MU] = NORMALIZE(X) also returns a vector MU of mean values for each
%   variable. 
%
%   [X MU D] = NORMALIZE(X) also returns a vector D containing the
%   Euclidean lengths for each original variable.
%
%   This function is an auxiliary part of SpAM, a matlab toolbox for
%   sparse modeling and analysis.
%
%  See also CENTER.

n = size(X,1);
[X mu] = center(X);
d = sqrt(sum(X.^2));
d(d == 0) = 1;
X = X./(ones(n,1)*d);
