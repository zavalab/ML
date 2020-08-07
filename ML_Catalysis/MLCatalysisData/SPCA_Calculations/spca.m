function [B SD L D paths] = spca(X, Gram, K, delta, stop, maxSteps, convergenceCriterion, verbose)
% SPCA  The sequential variant of the SPCA algorithm of Zou et. al [1] for
% computing sparse principal components. 
%
%    [B SV L D PATHS] = SPCA(X, GRAM, K, DELTA, STOP) computes sparse
%    principal components of the data in X. X is an n x p matrix where n is
%    the number of observations and p is the number of variables. X should
%    be centered and normalized such that the column means are 0 and the
%    column Euclidean lengths are 1. Gram = X'*X is the p x p Gram matrix.
%    Either X, Gram or both must be supplied. Pass an empty matrix as
%    argument if either X or Gram is missing.
%    K is the desired number of sparse principal components.
%    DELTA specifies the positive ridge (L2) term coefficient. If DELTA
%    is set to infinity, soft thresholding is used to calculate the
%    components. This is appropriate when p>>n and results in a
%    significantly faster algorithm.
%    STOP is the stopping criterion. If STOP is negative, its absolute
%    (integer) value corresponds to the desired number of non-zero
%    variables. If STOP is positive, it corresponds to an upper bound on
%    the L1-norm of the BETA coefficients. STOP = 0 results in a regular
%    PCA. Supply either a single STOP value, or a vector of K STOP values,
%    one for each component.
%
%    SPCA(X, GRAM, K, DELTA, STOP, MAXSTEPS) sets the maximum number of
%    iterations before the estimation of each component is terminated.
%    Default is MAXSTEPS = 300. 
%
%    SPCA(X, GRAM, K, DELTA, STOP, MAXSTEPS, CONVERGENCECRITERION)
%    specifies a threshold on the difference (in the squared two-norm 
%    sense) between the currently estimated sparse loading vector between
%    iterations. When the difference falls below this threshold the
%    algorithm is said to have  converged. Default is CONVERGENCECRITERION
%    = 1e-9. 
%
%    SPCA(X, GRAM, K, DELTA, STOP, MAXSTEPS, CONVERGENCECRITERION,
%    VERBOSE) with VERBOSE set to true will turn on display of algorithm
%    information. Default is VERBOSE = false.
%
%    SPCA returns B, the sparse loading vectors (principal component
%    directions); SV, the adjusted variances of each sparse component; L
%    and D, the loadings and variances of regular PCA; and PATHS, a struct
%    containing the loading paths for each component as functions of
%    iteration number.
%
%    Note that if X is omitted, the absolute values of SV cannot be
%    trusted, however, the relative values will still be correct.
%
%    Example
%    -------
%    Compare PCA and SPCA on a data set with three latent components, one
%    step edge, one component with a single centered Gaussian and one
%    component with two Gaussians spread apart.
%
%    % Fix stream of random numbers
%    s1 = RandStream.create('mrg32k3a','Seed', 11);
%    s0 = RandStream.setDefaultStream(s1);
%    % Create synthetic data set
%    n = 1500; p = 500;
%    t = linspace(0, 1, p);
%    pc1 = max(0, (t - 0.5)> 0);
%    pc2 = 0.8*exp(-(t - 0.5).^2/5e-3);
%    pc3 = 0.4*exp(-(t - 0.15).^2/1e-3) + 0.4*exp(-(t - 0.85).^2/1e-3);
%    X = [ones(n/3,1)*pc1 + randn(n/3,p); ones(n/3,1)*pc2 + ...
%      randn(n/3,p); ones(n/3,1)*pc3 + randn(n/3,p)];
%    % PCA and SPCA
%    [U D V] = svd(X, 'econ');
%    d = sqrt(diag(D).^2/n);
%    [B SD] = spca(X, [], 3, inf, -[250 125 100], 3000, 1e-3, true);
%    figure(1)
%    plot(t, [pc1; pc2; pc3]); axis([0 1 -1.2 1.2]);
%    title('Noiseless data');
%    figure(2);
%    plot(t, X);  axis([0 1 -6 6]);
%    title('Data + noise');
%    figure(3);
%    plot(t, d(1:3)*ones(1,p).*(V(:,1:3)'));  axis([0 1 -1.2 1.2]);
%    title('PCA');
%    figure(4)
%    plot(t, sqrt(SD)*ones(1,p).*(B'));  axis([0 1 -1.2 1.2]);
%    title('SPCA');
%    % Restore random stream
%    RandStream.setDefaultStream(s0);
%
%    References
%    -------
%    [1] H. Zou, T. Hastie, and R. Tibshirani. Sparse Principal Component
%    Analysis. J. Computational and Graphical Stat. 15(2):265-286, 2006.
%
%  See also ELASTICNET.

%% Input checking and initialization
if nargin < 8
  verbose = 0;
end
if nargin < 7
  convergenceCriterion = 1e-9;
end
if nargin < 6
  maxSteps = 300;
end
if nargin < 5
  error('SpaSM:spca', 'Minimum five arguments are required');
end
if nargout == 5
  storepaths = 1;
else
  storepaths = 0;
end
if isempty(X) && isempty(Gram)
  error('SpaSM:spca', 'Must supply a data matrix or a Gram matrix or both.');
end

%% SPCA algorithm setup

if isempty(X)
  % Infer X from X'*X
  [Vg Dg] = eig(Gram);
  X = Vg*sqrt(abs(Dg))*Vg';
end

[n p] = size(X);

% Number of sparse loading vectors / principal components
K = min([K p n-1]);

% Standard PCA (starting condition for SPCA algorithm)
[FOO, S, L] = svd(X, 'econ');
D = diag(S).^2/n; % PCA variances

% Replicate STOP value for all components if necessary
if length(stop) ~= K
  stop = stop(1)*ones(1,K);
end

% allocate space for loading paths
if storepaths
  paths(1:K) = struct('loadings', []);
end

% setup SPCA matrices A and B
A = L(:,1:K);
B = zeros(p,K);

%% SPCA loop
% for each component
for k = 1:K
  
  step = 0; % current algorithm iteration number
  converged = false;
  
  if verbose
    disp(['Estimating component ' num2str(k)]);
  end
    
  while ~converged && step < maxSteps
    step = step + 1;
    
    Bk_old = B(:,k);
    if delta == inf
      % Soft thresholding, calculate beta directly
      if isempty(Gram)
        AXX = (A(:,k)'*X')*X;
      else
        AXX = A(:,k)'*Gram;
      end
      if stop(k) < 0 && -stop(k) < p
        sortedAXX = sort(abs(AXX), 'descend');
        B(:,k) = ( sign(AXX).*max(0, abs(AXX) - sortedAXX(-floor(stop(k)) + 1)) )';
      else
        B(:,k) = ( sign(AXX).*max(0, abs(AXX) - stop(k)) )';
      end
    else
      % Find beta by elastic net regression
      B(:,k) = larsen(X, X*A(:,k), delta, stop(k), Gram, false, false);
    end
    
    % Normalize coefficients such that loadings has Euclidean length 1
    B_norm = sqrt(B(:,k)'*B(:,k));
    if B_norm == 0
      B_norm = 1;
    end
    B(:,k) = B(:,k)/B_norm;
    
    % converged?
    criterion = sum((Bk_old - B(:,k)).^2);
    if verbose && ~mod(step, 10)
      disp(['  Iteration: ' num2str(step) ', convergence criterion: ' num2str(criterion)]);
    end    
    converged = criterion < convergenceCriterion;
    
    % Save loading path data
    if storepaths
      paths(k).loadings = [paths(k).loadings B(:,k)];
    end
    
    % Update A
    if isempty(Gram)
      t = X'*(X*B(:,k));
    else
      t = Gram*B(:,k);
    end
    S = t - A(:,1:k-1)*(A(:,1:k-1)'*t);
    A(:,k) = S/sqrt(S'*S); % normalize to unit length
  end
  
  if verbose
    disp(['  Iteration: ' num2str(step) ', convergence criterion: ' num2str(criterion)]);
  end
    
end

%% Calculate adjusted variances
if K == 1
  SD = sum((X*B).^2)/n;
else
  SD = diag(qr(X*B,0)).^2/n;
end

%% Print information
if verbose
  if p < 20
    fprintf('\n\n --- Sparse loadings ---\n');
    disp(B)
  end
  fprintf('\n --- Adjusted variances, Variance of regular PCA ---\n');
  disp([SD/sum(D) D(1:K)/sum(D)])
  fprintf('Total: %3.2f%% %3.2f%%', 100*sum(SD/sum(D)), 100*sum(D(1:K)/sum(D)));
  fprintf('\nNumber of nonzero loadings:');
  disp(sum(abs(B) > 0));
end

