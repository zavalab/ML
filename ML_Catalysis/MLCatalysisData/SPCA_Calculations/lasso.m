function [b info] = lasso(X, y, stop, storepath, verbose)
%LASSO The LARS algorithm for estimating LASSO solutions for various values
%of the regularization parameter.
%
%   BETA = LASSO(X, Y) performs LASSO [1] regression on the variables in
%   X to approximate the response y. Variables X (n x p) are assumed to be
%   normalized (zero mean, unit length), the response y (n x 1) is assumed
%   to be centered. BETA contains the LASSO solutions for all values of the
%   regularization parameter starting from a large value corresponding to
%   all beta coefficients equal to zero, to lambda = 0, corresponding to
%   the ordinary least squares solution.
%   BETA = LASSO(X, Y, STOP) with nonzero STOP will estimate lasso
%   solutions with early stopping. If STOP is negative, STOP is an integer
%   that determines the desired number of non-zero variables. If STOP is
%   positive, it corresponds to an upper bound on the L1-norm of the BETA
%   coefficients. Setting STOP to zero (default) yields the entire
%   regularization path.
%   BETA = LASSO(X, Y, STOP, STOREPATH) with STOREPATH set to true will
%   return the entire regularization path from zero active variables (a
%   high value of lambda) to the point where the STOP criterion is met, or
%   the least squares solution in reached when STOP = 0. Setting STOREPATH
%   to false will yield the single solution at which the STOP criterion is
%   met, thus saving precious computer resources.
%   BETA = LASSO(X, Y, STOP, STOREPATH, VERBOSE) with VERBOSE set to true
%   will print the adding and subtracting of variables as LASSO solutions
%   are found.
%   [BETA INFO] = LASSO(...) returns a structure INFO containing various
%   useful measures. If the entire solution path has been calculated, INFO
%   containts the goodness-of-fit estimates AIC (Akaike's Information
%   Criterion), BIC (Bayesian Information Criterion), Mallow's Cp
%   statistic, and the number of degrees of freedom at each step. It also
%   includes the L1 penalty constraints s and lambda of which the former
%   comes from the formulation beta = arg min ||y - Xb||^2 subject to
%   ||beta||_1<=t and s is t defined in the range [0,1], and the latter is
%   lambda in the forumlation beta = arg min ||y - X*beta||^2 +
%   lambda*||beta||_1. If stop <> 0 and a single LASSO solution is
%   returned, INFO will only contain the lambda and s value at the solution
%   along with the number of degrees of freedom. INFO also includes the
%   number of steps made to compute the solution, including the first step
%   where beta is the zero vector.
%
%   The algorithm is a variant of the LARS algorithm [2] with elements from
%   [3].
%
%   Example
%   -------
%   Compute the full LASSO path from a synthetic data set with six
%   variables where the model is a linear combination of the first three
%   variables. Visualize the path along with the best solution in terms of
%   the AIC goodness-of-fit measure.
%
%   % Fix stream of random numbers
%   s1 = RandStream.create('mrg32k3a','Seed', 22);
%   s0 = RandStream.setDefaultStream(s1);
%   % Create data set
%   n = 100; p = 6;
%   correlation = 0.6;
%   Sigma = correlation*ones(p) + (1 - correlation)*eye(p);
%   mu = zeros(p,1);
%   X = mvnrnd(mu, Sigma, n);
%   % Model is lin.comb. of first three variables plus noise
%   y = X(:,1) + X(:,2) + X(:,3) + 2*randn(n,1);
%   % Preprocess data
%   X = normalize(X);
%   y = center(y);
%   % Run LASSO
%   [beta info] = lasso(X, y, 0, true, true);
%   % Find best fitting model
%   [bestAIC bestIdx] = min(info.AIC);
%   best_s = info.s(bestIdx);
%   % Plot results
%   h1 = figure(1);
%   plot(info.s, beta, '.-');
%   xlabel('s'), ylabel('\beta', 'Rotation', 0)
%   line([best_s best_s], [-6 14], 'LineStyle', ':', 'Color', [1 0 0]);
%   legend('1','2','3','4','5','6',2);
%   % Restore random stream
%   RandStream.setDefaultStream(s0);
%
%   References
%   -------
%   [1] R. Tibshirani. Regression shrinkage and selection via the lasso. J.
%   Royal Statist. Soc. B., 58(1):267-288, 1996.
%   [2] B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani. Least Angle
%   Regression. Ann. Statist. 32(2):407-499, 2004.
%   [3] S. Rosset, and Ji Zhu. Piecewise Linear Regularized Solution Paths.
%   Ann. Statist. 35(3):1012-1030, 2007.
%
%  See also LAR, ELASTICNET.

%% Input checking
% Set default values.
if nargin < 5
  verbose = false;
end
if nargin < 4
  storepath = true;
end
if nargin < 3
  stop = 0;
end
if nargin < 2
  error('SpaSM:lasso', 'Input arguments X and y must be specified.');
end

%% LARS variable setup
[n p] = size(X);

Gram = [];
% if n is approximately a factor 10 bigger than p it is faster to use a
% precomputed Gram matrix rather than Cholesky factorization when solving
% the partial OLS problem. Make sure the resulting Gram matrix is not
% prohibitively large.
if (n/p) > 10 && p < 1000
  Gram = X'*X;
end

%% Run the LARS algorithm
[b steps] = larsen(X, y, 0, stop, Gram, storepath, verbose);

%% Compute auxilliary measures
if nargout == 2 % only compute if asked for
  info.steps = steps;
  b0 = pinv(X)*y; % regression coefficients of low-bias model
  penalty0 = sum(abs(b0)); % L1 constraint size of low-bias
  indices = (1:p)';
  
  if storepath % for entire path
    q = info.steps + 1;
    sigma2e = sum((y - X*b0).^2)/n;
    info.lambda = zeros(1,q);
    info.df = zeros(1,q);
    info.Cp = zeros(1,q);
    info.AIC = zeros(1,q);
    info.BIC = zeros(1,q);
    info.s = zeros(1,q);
    for step = 1:q
      A = indices(b(:,step) ~= 0); % active set
      % compute godness of fit measurements Cp, AIC and BIC
      r = y - X(:,A)*b(A,step); % residuals
      rss = sum(r.^2); % residual sum-of-squares
      info.df(step) = length(A);
      info.Cp(step) = rss/sigma2e - n + 2*info.df(step);
      info.AIC(step) = rss + 2*sigma2e*info.df(step);
      info.BIC(step) = rss + log(n)*sigma2e*info.df(step);
      % compute L1 penalty constraints s and lambda
      info.s(step) = sum(abs(b(A,step)))/penalty0;
      if (step == 1)
        info.lambda(step) = max(2*abs(X'*y));
      else
        info.lambda(step) = median(2*abs(X(:,A)'*r));
      end
    end
    
  else % for single solution
    % compute L1 penalty constraints s and lambda at solution
    A = indices(b ~= 0); % active set
    info.s = sum(abs(b))/penalty0;
    info.df = length(A);
    info.steps = steps;
    if isempty(A)
      info.lambda = max(2*abs(X'*y));
    else
      info.lambda = median(2*abs(X(:,A)'*(y - X(:,A)*b(A))));
    end
  end
  
end
