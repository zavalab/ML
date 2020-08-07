function [b info] = elasticnet(X, y, delta, stop, storepath, verbose)
%ELASTICNET Regularization and variable selection for regression via the
%   Elastic Net.
%
%   BETA = ELASTICNET(X, Y, DELTA) performs Elastic Net [1] regression on
%   the variables in X to approximate the response y. Variables X (n x p)
%   are assumed to be normalized (zero mean, unit length), the response y
%   (n x 1) is assumed to be centered. The regularization parameter DELTA
%   specifies the weight on the L2 penalty on the regression coefficients.
%   A positive value of DELTA makes analyses of p>n datasets possible and
%   encourages grouping of correlated variables. Setting DELTA to zero
%   yields the LASSO solution. BETA contains the Elastic Net solutions for
%   all values of the L1 regularization parameter, starting from a large
%   value corresponding to all beta coefficients equal to zero, to 0,
%   corresponding to the ordinary least squares solution.
%   BETA = ELASTICNET(X, Y, DELTA, STOP) with nonzero STOP yields Elastic
%   Net solutions with early stopping at a particular value of the L1
%   regularization parameter. If STOP is negative, STOP is an integer that
%   determines the desired number of non-zero variables. If STOP is
%   positive, it corresponds to an upper bound on the L1-norm of the BETA
%   coefficients. Setting STOP to zero (default) yields the entire
%   regularization path.
%   BETA = ELASTICNET(X, Y, DELTA, STOP, STOREPATH) with STOREPATH set to
%   true (default) will return the entire regularization path from zero
%   active variables (a high value of delta) to the point where the STOP
%   criterion is met, or the least squares solution in reached when STOP =
%   0. Setting STOREPATH to false will yield the single solution at which
%   the STOP criterion is met, thus saving precious computer resources.
%   BETA = ELASTICNET(X, Y, STOP, STOREPATH, VERBOSE) with VERBOSE set to
%   true (default is false) will print the adding and subtracting of
%   variables as LASSO solutions are found.
%   [BETA INFO] = ELASTICNET(...) returns a structure INFO containing
%   various useful measures. If the entire solution path has been
%   calculated, INFO containts the goodness-of-fit estimates AIC (Akaike's
%   Information Criterion), BIC (Bayesian Information Criterion), Mallow's
%   Cp statistic, and the number of degrees of freedom at each step. It
%   also includes the L1 penalty constraints s and lambda of which the
%   former represents the size of the L1 constraint defined in the range
%   [0,1], and the latter is lambda in the forumlation beta = arg min ||y
%   - X*beta||^2 + delta*||beta||^2 + lambda*||beta||_1. If stop <> 0
%   and a single Elastic Net solution is returned, INFO will only contain
%   the lambda and s value at the solution along with the number of
%   degrees of freedom. INFO also includes the number of steps made to
%   compute the solution, including the first step where beta is the zero
%   vector.
%
%   The algorithm is a variant of the LARS-EN algorithm [1].
%
%   Example
%   -------
%   Compute the full Elastic Net solution path from a synthetic data set
%   with 40 variables where the model is a linear combination of the first
%   three variables plus noise. Visualize the path, where the the three
%   variables included in the model can be distinguished from the rest.
%
%   % Fix stream of random numbers
%   s1 = RandStream.create('mrg32k3a','Seed', 42);
%   s0 = RandStream.setDefaultStream(s1);
%   % Create data set
%   n = 30; p = 40;
%   correlation = 0.2;
%   Sigma = correlation*ones(p) + (1 - correlation)*eye(p);
%   mu = zeros(p,1);
%   X = mvnrnd(mu, Sigma, n);
%   % Model is lin.comb. of first three variables plus noise
%   y = X(:,1) + X(:,2) + X(:,3) + 0.5*randn(n,1);
%   % Preprocess data
%   X = normalize(X);
%   y = center(y);
%   % Run LASSO
%   delta = 1e-3;
%   [beta info] = elasticnet(X, y, delta, 0, true, true);
%   % Plot results
%   h1 = figure(1);
%   plot(info.s, beta, '.-');
%   xlabel('s'), ylabel('\beta', 'Rotation', 0)
%   % Restore random stream
%   RandStream.setDefaultStream(s0);
%
%   References
%   -------
%   [1] H. Zou and T. Hastie. Regularization and variable selection via the
%   elastic net. J. Royal Stat. Soc. B. 67(2):301-320, 2005.
%
%  See also LAR, LASSO.

%% Input checking
% Set default values.
if nargin < 6
  verbose = false;
end
if nargin < 5
  storepath = true;
end
if nargin < 4
  stop = 0;
end
if nargin < 3
  error('SpaSM:elasticnet', 'Input arguments X, y and delta must be specified.');
end
if delta < 0
  error('SpaSM:elasticnet', 'Parameter delta must be positive.');
end

%% Elastic Net variable setup
[n p] = size(X);

Gram = [];
% if n is approximately a factor 10 bigger than p it is faster to use a
% precomputed Gram matrix rather than Cholesky factorization when solving
% the partial OLS problem. Make sure the resulting Gram matrix is not
% prohibitively large.
if (n/p) > 10 && p < 1000
  Gram = X'*X + delta*eye(p);
end

%% Calculate Elastic Net solutions with the LARS-EN algorithm
[b steps] = larsen(X, y, delta, stop, Gram, storepath, verbose);

% adjust to avoid double shrinkage (non-naïve Elastic Net solution)
b = (1 + delta)*b;

%% Compute auxilliary measures
if nargout == 2 % only compute if asked for
  info.steps = steps;
  if (delta < eps)
    % delta is zero, use minimum-norm solution
    b0 = pinv(X)*y;
  else
    % delta is non-zero, use ridge regression solution
    [U D V] = svd(X, 'econ');
    b0 = V*diag(1./(diag(D).^2 + delta))*D*U'*y;
  end
  penalty0 = sum(abs(b0)); % L1 constraint size of low-bias model
  indices = (1:p)';
  
  if storepath % for entire path
    q = info.steps + 1;
    info.lambda = zeros(1,q);
    info.df = zeros(1,q);
    info.Cp = zeros(1,q);
    info.AIC = zeros(1,q);
    info.BIC = zeros(1,q);
    info.s = zeros(1,q);
    sigma2e = sum((y - X*b0).^2)/n; % Mean Square Error of low-bias model
    for step = 1:q
      A = indices(b(:,step) ~= 0); % active set
      [U D] = svd(X(:,A), 'econ');
      d2 = diag(D).^2;
      info.df(step) = sum(sum(U.*(U*diag(d2./(d2 + delta)))));
      % compute godness of fit measurements Cp, AIC and BIC
      r = y - X(:,A)*b(A,step); % residuals
      rss = sum(r.^2); % residual sum-of-squares
      info.Cp(step) = rss/sigma2e - n + 2*info.df(step);
      info.AIC(step) = rss + 2*sigma2e*info.df(step);
      info.BIC(step) = rss + log(n)*sigma2e*info.df(step);
      % compute L1 penalty constraints s and lambda
      info.s(step) = sum(abs(b(A,step)))/penalty0;
      if (step == 1)
        info.lambda(step) = max(2*abs(X'*y));
      else
        r2 = y - X(:,A)*b(A,step)/(1 + delta);
        info.lambda(step) = median(2*abs(X(:,A)'*r2 - delta/(1 + delta)*b(A,step)));
      end
    end
    
  else % for single solution
    info.steps = steps;
    A = indices(b ~= 0); % active set
    % compute L1 penalty constraints s and lambda at solution
    info.s = sum(abs(b))/penalty0;
    [U D] = svd(X(:,A), 'econ');
    d2 = diag(D).^2;
    info.df = sum(sum(U.*(U*diag(d2./(d2 + delta)))));
    if isempty(A)
      info.lambda = max(2*abs(X'*y));
    else
      r2 = y - X(:,A)*b(A)/(1 + delta);
      info.lambda = median(2*abs(X(:,A)'*r2 - delta/(1 + delta)*b(A)));
    end
  end
  
end
