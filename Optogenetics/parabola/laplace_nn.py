import numpy as np
import jax.numpy as jnp
from jax import nn, jacfwd, jacrev, jit, vmap, lax, random
from functools import partial
import time

# import scipy's optimizer
from scipy.optimize import minimize

# import matrix math functions
from .linalg import *


class FFNN():

    def __init__(self, n_inputs, n_hidden, n_outputs, param_0=.2):

        # store dimensions
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # bounds on initial parameter guess
        self.param_0 = param_0

        # determine shapes of weights/biases = [Wih, bih, Who, bho]
        self.shapes = [[n_hidden, n_inputs], [n_hidden], [n_outputs, n_hidden], [n_outputs]]
        self.k_params = []
        self.n_params = 0
        for shape in self.shapes:
            self.k_params.append(self.n_params)
            self.n_params += np.prod(shape)
        self.k_params.append(self.n_params)

        # initialize parameters
        self.params = np.zeros(self.n_params)
        for k1, k2, shape in zip(self.k_params[:-1], self.k_params[1:-1], self.shapes[:-1]):
            self.params[k1:k2] = np.random.uniform(-self.param_0, self.param_0, k2 - k1)

        # initialize hyper-parameters
        self.a = 1e-4
        self.b = 1e-4

        # initialize covariance
        self.Ainv = None

        ### define jit compiled functions ###

        # batch prediction
        self.forward_batch = jit(vmap(self.forward, (None, 0)))

        # jit compile gradient w.r.t. params
        self.Gi = jit(jacfwd(self.forward))
        self.G = jit(jacfwd(self.forward_batch))

        # jit compile function to compute gradient of loss w.r.t. parameters
        self.compute_grad_NLL = jit(jacrev(self.compute_NLL))

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        # params is a vector = [Wih, bih, Who, bho]
        return [np.reshape(params[k1:k2], shape) for k1, k2, shape in
                zip(self.k_params, self.k_params[1:], self.shapes)]

    # per-sample prediction
    @partial(jit, static_argnums=0)
    def forward(self, params, sample):
        # reshape params
        Wih, bih, Who, bho = self.reshape(params)

        # hidden layer
        h = nn.tanh(Wih @ sample + bih)

        # output
        out = Who @ h + bho

        return out

    # estimate posterior parameter distribution
    def fit(self, X, Y, evd_tol=1e-3, nlp_tol=None, alpha_0=1e-5, alpha_1=1., patience=1, max_fails=3):

        # estimate parameters using gradient descent
        self.itr = 0
        passes = 0
        fails = 0
        convergence = np.inf
        previdence = -np.inf

        # init convergence status
        converged = False

        # initialize hyper parameters
        self.init_hypers(X, Y, alpha_0)

        while not converged:
            # update Alpha and Beta hyper-parameters
            if self.itr > 0: self.update_hypers(X, Y)

            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective,
                                jac=self.jacobian,
                                hess=self.hessian,
                                x0=self.params,
                                args=(X, Y,),
                                tol=nlp_tol,
                                method='Newton-CG',
                                callback=self.callback)
            self.params = self.res.x
            self.loss = self.res.fun

            # update parameter precision matrix (Hessian)
            print("Updating precision...")
            if self.itr == 0:
                self.alpha = alpha_1 * jnp.ones_like(self.params)
            self.update_precision(X, Y)

            # update evidence
            self.update_evidence()
            print("Evidence {:.3f}".format(self.evidence))

            # check convergence
            convergence = np.abs(previdence - self.evidence) / np.max([1., np.abs(self.evidence)])

            # update pass count
            if convergence < evd_tol:
                passes += 1
                print("Pass count ", passes)
            else:
                passes = 0

            # increment fails if convergence is negative
            if self.evidence < previdence:
                fails += 1
                print("Fail count ", fails)

            # finally compute covariance (Hessian inverse)
            self.update_covariance(X, Y)

            # determine whether algorithm has converged
            if passes >= patience:
                converged = True

            # terminate if maximum number of mis-steps exceeded
            # if fails >= max_fails:
            #     print(
            #         "Warning: Exceeded max number of attempts to increase model evidence, model could not converge to specified tolerance.")
            #     converged = True

            # update evidence
            previdence = np.copy(self.evidence)
            self.itr += 1

            # return

    def callback(self, xk, res=None):
        print("Loss: {:.3f}".format(self.loss))
        return True

    # function to compute NLL loss function
    @partial(jit, static_argnums=(0,))
    def compute_NLL(self, params, X, Y, Beta):
        outputs = self.forward_batch(params, X)
        error = jnp.nan_to_num(outputs - Y)
        return jnp.einsum('nk,kl,nl->', error, Beta, error) / 2.

    # define objective function
    def objective(self, params, X, Y):
        # init loss with parameter penalty
        self.loss = jnp.dot(self.alpha * params, params) / 2.

        # forward pass
        self.loss += self.compute_NLL(params, X, Y, self.Beta)

        return self.loss

    # define function to compute gradient of loss w.r.t. parameters
    def jacobian(self, params, X, Y):

        # gradient of -log prior
        g = self.alpha * params

        # gradient of -log likelihood
        g += self.compute_grad_NLL(params, X, Y, self.Beta)

        # return gradient of -log posterior
        return g

    # define function to compute approximate Hessian
    def hessian(self, params, X, Y):
        # init w/ hessian of -log(prior)
        A = jnp.diag(self.alpha)

        # outer product approximation of Hessian:

        # Compute gradient of model output w.r.t. parameters
        G = self.G(params, X)

        # update Hessian
        A += A_next(G, self.Beta)

        return (A + A.T)/2.

    # update hyper-parameters alpha and Beta
    def init_hypers(self, X, Y, alpha_0):
        # compute number of independent samples in the data
        self.N = X.shape[0] 

        # init alpha
        self.alpha = alpha_0 * jnp.ones_like(self.params)

        # update Beta
        self.Beta = jnp.eye(self.n_outputs)
        self.BetaInv = jnp.eye(self.n_outputs)

    # update hyper-parameters alpha and Beta
    def update_hypers(self, X, Y):

        # compute measurement covariance
        yCOV = 0.

        # forward
        outputs = self.forward_batch(self.params, X)
        error = jnp.nan_to_num(outputs - Y)

        # backward
        G = self.G(self.params, X)

        # update measurement covariance
        yCOV += compute_yCOV(error, G, self.Ainv)

        # make sure prediction covariance is symmetric
        yCOV = (yCOV + yCOV.T) / 2.
        # update alpha
        self.alpha = 1. / (self.params ** 2 + jnp.diag(self.Ainv) + 2. * self.a)
        # alpha = self.n_params / (jnp.sum(self.params**2) + jnp.trace(self.Ainv) + 2.*self.a)
        # self.alpha = alpha*jnp.ones_like(self.params)

        # update beta
        self.Beta = self.N * jnp.linalg.inv(yCOV + 2. * self.b * jnp.eye(self.n_outputs))
        self.Beta = make_pos_def((self.Beta + self.Beta.T) / 2., jnp.ones(self.n_outputs))
        self.BetaInv = jnp.linalg.inv(self.Beta)

    # compute precision matrix
    def update_precision(self, X, Y):

        # compute inverse precision (covariance Matrix)
        A = np.diag(self.alpha)

        # update A
        G = self.G(self.params, X)
        A += A_next(G, self.Beta)

        # make sure that matrices are symmetric and positive definite
        self.A = make_pos_def((A + A.T) / 2., self.alpha)

    # compute covariance matrix
    def update_covariance(self, X, Y):
        ### fast / approximate method: ###
        self.Ainv = make_pos_def(compute_Ainv(self.A), jnp.ones(self.n_params))

    # compute the log marginal likelihood
    def update_evidence(self):
        # compute evidence
        self.evidence = self.N / 2 * log_det(self.Beta) + \
                        1 / 2 * np.nansum(np.log(self.alpha)) - \
                        1 / 2 * log_det(self.A) - self.loss

    # function to predict mean of outcomes
    def predict_point(self, X):
        # make point predictions
        preds = jnp.clip(self.forward_batch(self.params, X), 0., 1.)

        return preds

    # function to predict mean and stdv of outcomes
    def predict(self, X):

        # function to get diagonal of a tensor
        get_diag = vmap(jnp.diag, (0,))

        # point estimates
        preds = np.array(self.predict_point(X))

        # compute sensitivities
        G = self.G(self.params, X)

        # compute covariances
        COV = np.array(compute_predCOV(self.BetaInv, G, self.Ainv))

        # pull out standard deviations
        stdvs = np.sqrt(get_diag(COV))

        return preds, stdvs

    # function to predict mean and stdv of outcomes given updated covariance
    def conditional_predict(self, X, Ainv):

        # function to get diagonal of a tensor
        get_diag = vmap(jnp.diag, (0,))

        # point estimates
        preds = self.predict_point(X)

        # compute sensitivities
        G = self.G(self.params, X)

        # compute covariances
        COV = compute_epistemic_COV(G, Ainv)

        # pull out standard deviations
        stdvs = jnp.sqrt(get_diag(COV))

        return preds, stdvs
