function [b info] = forwardselection(X, y, stop, storepath, verbose)
%FORWARDSELECTION A simple forward selection regression method to be used
%as baseline for other methods in the SpaSM toolbox.
%
%   BETA = FORWARDSELECTION(X, y) performs forward selection (FS) on the
%   variables in X to approximate the response y. Variables X (n x p) are
%   assumed to be normalized (zero mean, unit length), the response y (n x
%   1) is assumed to be centered. BETA contains all FS solutions starting
%   from zero active variables to the ordinary least squares solution when
%   n > p or the point where n - 1 variables are active for problems where
%   n <= p.
%   BETA = FS(X, Y, STOP) with nonzero STOP will estimate FS solutions
%   with early stopping. If STOP is negative, STOP is an integer that
%   determines the desired number of variables. If STOP is positive, it
%   corresponds to an upper bound on the L1-norm of the BETA coefficients.
%   BETA = FS(X, Y, STOP, STOREPATH) with STOREPATH set to true will
%   return the entire path from zero active variables to the point where
%   the STOP criterion is met, or the complete path for STOP = 0. Setting
%   STOREPATH to false will yield the single solution at which the STOP
%   criterion is met, thus saving precious computer resources.
%   BETA = FS(X, Y, STOP, STOREPATH, VERBOSE) with VERBOSE set to true
%   will print the adding of variables as FS solutions are found.
%   [BETA INFO] = FS(...) returns a structure INFO containing various
%   useful measures. If the entire solution path has been calculated, INFO
%   containts the goodness-of-fit estimates AIC (Akaike's Information
%   Criterion), BIC (Bayesian Information Criterion) and Mallow's Cp
%   statistic. It also includes a variable s defined as the L1 size of beta
%   at each step in the range [0,1]. If stop <> 0 and a single FS solution
%   is returned, INFO will only the s value at the solution. INFO also
%   includes the number of degrees of freedom at each step and the total
%   number of steps made to compute the solution, including the first step
%   where beta is the zero vector.
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
%   [beta info] = forwardselection(X, y, 0, true, true);
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
%  See also LAR, LASSO, ELASTICNET.

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
  error('SpaSM:forwardselection', 'Input arguments X and y must be specified.');
end

%% Forward selection variable setup
[n p] = size(X);
maxVariables = min(n-1,p); % Maximum number of active variables

useGram = false;
% if n is approximately a factor 10 bigger than p it is faster to use a
% precomputed Gram matrix rather than Cholesky factorization when solving
% the partial OLS problem. Make sure the resulting Gram matrix is not
% prohibitively large.
if (n/p) > 10 && p < 1000
  useGram = true;
  Gram = X'*X;
end

% set up the FS coefficient vector
if storepath
  b = zeros(p, 2*p);
else
  b = zeros(p, 1);
  b_prev = b;
end

mu = zeros(n, 1); % current "position" towards full lsq solution

I = 1:p; % inactive set
A = []; % active set
if ~useGram
  R = []; % Cholesky factorization R'R = X'X where R is upper triangular
end

stopCond = 0; % Early stopping condition boolean
step = 1; % step count

if verbose
  fprintf('Step\tAdded\tActive set size\n');
end

%% Forward selection main loop
% while not at OLS solution or early stopping criterion is met
while length(A) < maxVariables && ~stopCond
  r = y - mu;
  
  % find max correlation
  c = X(:,I)'*r;
  [cmax cidx] = max(abs(c));
  
  % add variable
  if ~useGram
    R = cholinsert(R,X(:,I(cidx)),X(:,A));
  end
  if verbose
    fprintf('%d\t\t%d\t\t%d\n', step, I(cidx), length(A) + 1);
  end
  A = [A I(cidx)];
  I(cidx) = [];
  
  % partial OLS solution and direction from current position to the OLS
  % solution of X_A
  if useGram
    b_OLS = Gram(A,A)\(X(:,A)'*y); % same as X(:,A)\y, but faster
  else
    b_OLS = R\(R'\(X(:,A)'*y)); % same as X(:,A)\y, but faster
  end
  
  % update beta
  if storepath
    b(A,step + 1) = b_OLS; % update beta
  else
    b_prev = b;
    b(A) = b_OLS; % update beta
  end
  
  % update position
  mu = X(:,A)*b_OLS;
  
  % increment step counter
  step = step + 1;
  
  % Early stopping at specified bound on L1 norm of beta
  if stop > 0
    if storepath
      t2 = sum(abs(b(:,step)));
      if t2 >= stop
        t1 = sum(abs(b(:,step - 1)));
        s = (stop - t1)/(t2 - t1); % interpolation factor 0 < s < 1
        b(:,step) = b(:,step - 1) + s*(b(:,step) - b(:,step - 1));
        stopCond = 1;
      end
    else
      t2 = sum(abs(b));
      if t2 >= stop
        t1 = sum(abs(b_prev));
        s = (stop - t1)/(t2 - t1); % interpolation factor 0 < s < 1
        b = b_prev + s*(b - b_prev);
        stopCond = 1;
      end
    end
  end
  
  % Early stopping at specified number of variables
  if stop < 0
    stopCond = length(A) >= -stop;
  end
end

% trim beta
if storepath && size(b,2) > step
  b(:,step + 1:end) = [];
end

%% Compute auxilliary measures
if nargout == 2 % only compute if asked for
  info.steps = step - 1;
  b0 = pinv(X)*y; % regression coefficients of low-bias model
  penalty0 = sum(abs(b0)); % L1 constraint size of low-bias model
  indices = (1:p)';
  
  if storepath % for entire path
    q = info.steps + 1;
    info.df = zeros(1,q);
    info.Cp = zeros(1,q);
    info.AIC = zeros(1,q);
    info.BIC = zeros(1,q);
    info.s = zeros(1,q);
    sigma2e = sum((y - X*b0).^2)/n;
    for step = 1:q
      A = indices(b(:,step) ~= 0); % active set
      % compute godness of fit measurements Cp, AIC and BIC
      r = y - X(:,A)*b(A,step); % residuals
      rss = sum(r.^2); % residual sum-of-squares
      info.df(step) = step - 1;
      info.Cp(step) = rss/sigma2e - n + 2*info.df(step);
      info.AIC(step) = rss + 2*sigma2e*info.df(step);
      info.BIC(step) = rss + log(n)*sigma2e*info.df(step);
      info.s(step) = sum(abs(b(A,step)))/penalty0;
    end
    
  else % for single solution
    info.s = sum(abs(b))/penalty0;
    info.df = info.steps - 1;
  end
  
end
