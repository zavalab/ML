import numpy as np
import jax.numpy as jnp
from jax import nn, jacfwd, jit, vmap, lax, random
from jax.scipy.stats.norm import cdf, pdf
from functools import partial
import time

# # import MCMC library
# import numpyro
# import numpyro.distributions as dist
# from numpyro.infer import MCMC, NUTS, HMC

class FFNN():

    def __init__(self, n_inputs, n_hidden, n_outputs,
                 param_0=.2, alpha_0=1., rng_key=123):

        # set rng key
        rng_key = random.PRNGKey(rng_key)

        # store dimensions
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # bounds on initial parameter guess
        self.param_0 = param_0
        self.alpha_0 = alpha_0

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
        for k1,k2,shape in zip(self.k_params, self.k_params[1:], self.shapes):
            if len(shape)>1:
                stdv = 1/np.sqrt(shape[-1])
            # self.params[k1:k2] = np.random.uniform(-self.param_0, self.param_0, k2-k1)
            self.params[k1:k2] = random.uniform(rng_key, shape=(k2-k1,), minval=-stdv, maxval=stdv)

        # initialize hyper-parameters
        self.a = 1e-4
        self.b = 1e-4

        # initialize covariance
        self.Ainv = None

        ### define jit compiled functions ###

        # batch prediction
        self.forward_batch = jit(vmap(self.forward, in_axes=(None, 0)))

        # jit compile gradient w.r.t. params
        self.G  = jit(jacfwd(self.forward_batch))
        self.Gi = jit(jacfwd(self.forward))

        # jit compile Newton update direction computation
        def NewtonStep(G, g, alpha, Beta):
            # compute hessian
            A = jnp.diag(alpha) + jnp.einsum('nki,kl,nlj->ij', G, Beta, G)
            # solve for Newton step direction
            d = jnp.linalg.solve(A, g)
            return d
        self.NewtonStep = jit(NewtonStep)

        # jit compile inverse Hessian computation step
        def Ainv_next(G, Ainv, BetaInv):
            GAinv = G@Ainv
            Ainv_step = GAinv.T@jnp.linalg.inv(BetaInv + GAinv@G.T)@GAinv
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_next = Ainv_next

        # jit compile measurement covariance computation
        def compute_yCOV(errors, G, Ainv):
            return jnp.einsum('nk,nl->kl', errors, errors) + jnp.einsum('nki,ij,nlj->kl', G, Ainv, G)
        self.compute_yCOV = jit(compute_yCOV)

        # jit compile prediction covariance computation
        def compute_searchCOV(Beta, G, Ainv):
            return jnp.eye(Beta.shape[0]) + jnp.einsum("kl,nli,ij,nmj->nkm", Beta, G, Ainv, G)
        self.compute_searchCOV = jit(compute_searchCOV)

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        # params is a vector = [Wih, bih, Who, bho]
        return [np.reshape(params[k1:k2], shape) for k1,k2,shape in zip(self.k_params, self.k_params[1:], self.shapes)]

    # per-sample prediction
    def forward(self, params, sample):
        # reshape params
        Wih, bih, Who, bho = self.reshape(params)

        # hidden layer
        h = nn.tanh(Wih@sample + bih)

        # output
        out = Who@h + bho

        return out

    # fit to data
    def fit(self, X, Y, lr=1e-2, map_tol=1e-3, evd_tol=1e-3, patience=3, max_fails=3):
        passes = 0
        fails  = 0
        # fit until convergence of evidence
        previdence = -np.inf
        evidence_converged = False
        epoch = 0
        best_evidence_params = np.copy(self.params)
        best_params = np.copy(self.params)

        while not evidence_converged:

            # update hyper-parameters
            self.update_hypers(X, Y)

            # use Newton descent to determine parameters
            prev_loss = np.inf

            # fit until convergence of NLP
            converged = False
            while not converged:
                # forward passs
                outputs = self.forward_batch(self.params, X)
                errors  = np.nan_to_num(outputs - Y)
                residuals = np.sum(errors)/X.shape[0]

                # compute convergence of loss function
                loss = self.compute_loss(errors)
                convergence = (prev_loss - loss) / max([1., loss])
                if epoch%10==0:
                    print("Epoch: {}, Loss: {:.5f}, Residuals: {:.5f}, Convergence: {:5f}".format(epoch, loss, residuals, convergence))

                # stop if less than tol
                if abs(convergence) <= map_tol:
                    # set converged to true to break from loop
                    converged = True
                else:
                    # lower learning rate if convergence is negative
                    if convergence < 0:
                        lr /= 2.
                        # re-try with the smaller step
                        self.params = best_params - lr*d
                    else:
                        # update best params
                        best_params = np.copy(self.params)

                        # update previous loss
                        prev_loss = loss

                        # compute gradients
                        G = self.G(self.params, X)
                        g = np.einsum('nk,kl,nli->i', errors, self.Beta, G) + self.alpha*self.params

                        # determine Newton update direction
                        d = self.NewtonStep(G, g, self.alpha, self.Beta)

                        # update parameters
                        self.params -= lr*d

                        # update epoch counter
                        epoch += 1

            # Update Hessian estimation
            G = self.G(self.params, X)
            self.A, self.Ainv = self.compute_precision(G)

            # compute evidence
            evidence = self.compute_evidence(X, loss)

            # determine whether evidence is converged
            evidence_convergence = (evidence - previdence) / max([1., abs(evidence)])
            print("\nEpoch: {}, Evidence: {:.5f}, Convergence: {:5f}".format(epoch, evidence, evidence_convergence))

            # stop if less than tol
            if abs(evidence_convergence) <= evd_tol:
                passes += 1
                lr *= 2.
            else:
                if evidence_convergence < 0:
                    # reset :(
                    fails += 1
                    self.params = np.copy(best_evidence_params)
                    # Update Hessian estimation
                    G = self.G(self.params, X)
                    self.A, self.Ainv = self.compute_precision(G)

                    # reset evidence back to what it was
                    evidence = previdence
                    # lower learning rate
                    lr /= 2.
                else:
                    passes = 0
                    # otherwise, update previous evidence value
                    previdence = evidence
                    # update measurement covariance
                    self.yCOV = self.compute_yCOV(errors, G, self.Ainv)
                    # update best evidence parameters
                    best_evidence_params = np.copy(self.params)

            # If the evidence tolerance has been passed enough times, return
            if passes >= patience or fails >= max_fails:
                evidence_converged = True


    # update hyper-parameters alpha and Beta
    def update_hypers(self, X, Y):
        if self.Ainv is None:
            self.yCOV = np.einsum('nk,nl->kl', np.nan_to_num(Y), np.nan_to_num(Y))
            self.yCOV = (self.yCOV + self.yCOV.T)/2.
            # update alpha
            self.alpha = self.alpha_0*np.ones(self.n_params)
            # update Beta
            self.Beta = X.shape[0]*np.linalg.inv(self.yCOV + 2.*self.b*np.eye(self.n_outputs))
            self.Beta = (self.Beta + self.Beta.T)/2.
            self.BetaInv = np.linalg.inv(self.Beta)
        else:
            # update alpha
            self.alpha = 1. / (self.params**2 + np.diag(self.Ainv) + 2.*self.a)
            # update beta
            self.Beta = X.shape[0]*np.linalg.inv(self.yCOV + 2.*self.b*np.eye(self.n_outputs))
            self.Beta = (self.Beta + self.Beta.T)/2.
            self.BetaInv = np.linalg.inv(self.Beta)

    # compute loss
    def compute_loss(self, errors):
        return 1/2*(np.einsum('nk,kl,nl->', errors, self.Beta, errors) + np.dot(self.alpha*self.params, self.params))

    # compute Precision and Covariance matrices
    def compute_precision(self, G):
        # compute Hessian (precision Matrix)
        A = jnp.diag(self.alpha) + jnp.einsum('nki,kl,nlj->ij', G, self.Beta, G)
        A = (A + A.T)/2.

        # compute inverse precision (covariance Matrix)
        Ainv = jnp.diag(1./self.alpha)
        for Gn in G:
            Ainv -= self.Ainv_next(Gn, Ainv, self.BetaInv)
        return A, Ainv

    # compute the log marginal likelihood
    def compute_evidence(self, X, loss):
        # compute evidence
        Hessian_eigs = np.linalg.eigvalsh(self.A)
        evidence = X.shape[0]/2*np.nansum(np.log(np.linalg.eigvalsh(self.Beta))) + \
                   1/2*np.nansum(np.log(self.alpha)) - \
                   1/2*np.nansum(np.log(Hessian_eigs[Hessian_eigs>0])) - loss
        return evidence

    def fit_MCMC(self, X, Y, num_warmup=1000, num_samples=4000, rng_key=0):

        # define probabilistic model
        def pyro_model():

            # sample from Laplace approximated posterior
            w = numpyro.sample("w",
                               dist.MultivariateNormal(loc=self.params,
                                                       covariance_matrix=self.Ainv))

            # sample from zero-mean Gaussian prior with independent precision priors
            '''with numpyro.plate("params", self.n_params):
                alpha = numpyro.sample("alpha", dist.Exponential(rate=1e-4))
                w = numpyro.sample("w", dist.Normal(loc=0., scale=(1./alpha)**.5))'''

            # sample from zero-mean Gaussian prior with single precision prior
            '''alpha = numpyro.sample("alpha", dist.Exponential(rate=1e-4))
            w = numpyro.sample("w", dist.MultivariateNormal(loc=np.zeros(self.n_params),
                                                            precision_matrix=alpha*np.eye(self.n_params)))'''

            # output of neural network:
            preds = self.forward_batch(w, X)

            # sample model likelihood with max evidence precision matrix
            numpyro.sample("Y",
                           dist.MultivariateNormal(loc=preds, precision_matrix=self.Beta),
                           obs = Y)

        # init MCMC object with NUTS kernel
        kernel = NUTS(pyro_model, step_size=1.)
        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

        # warmup
        self.mcmc.warmup(random.PRNGKey(rng_key), init_params=self.params)

        # run MCMC
        self.mcmc.run(random.PRNGKey(rng_key), init_params=self.params)

        # save posterior params
        self.posterior_params = np.array(self.mcmc.get_samples()['w'])

    # function to predict metabolites and variance
    def predict(self, X):
        # make point predictions
        preds = self.forward_batch(self.params, X)

        # compute sensitivities
        G = self.G(self.params, X)

        # compute covariances
        COV = self.BetaInv + np.einsum("nki,ij,nlj->nkl", G, self.Ainv, G)

        # pull out standard deviations
        get_diag = vmap(jnp.diag, (0,))
        stdvs = np.sqrt(get_diag(COV))

        return preds, stdvs, COV

    # function to return predicted mean
    def predict_point(self, X):
        # make point predictions
        preds = self.forward_batch(self.params, X)

        return preds

    # function to predict metabolites and variance
    def conditional_predict(self, X, X_design):
        # make point predictions
        preds = self.forward_batch(self.params, X)

        # compute sensitivities
        G = self.G(self.params, X)

        # compute sensitivities to design points
        G_design = self.G(self.params, X_design)

        # compute conditional parameter covariance
        Ainv = self.Ainv.copy()
        for Gn in G_design:
            Ainv -= self.Ainv_next(Gn, Ainv, self.BetaInv)

        # compute updated *epistemic* prediction covariance
        COV = self.BetaInv + np.einsum("nki,ij,nlj->nkl", G, Ainv, G)

        # pull out standard deviations
        get_diag = vmap(jnp.diag, (0,))
        stdvs = np.sqrt(get_diag(COV))

        return preds, stdvs, COV

    # function to predict metabolites and variance
    def conditioned_stdv(self, X, Ainv):

        # compute sensitivities
        G = self.G(self.params, X)

        # compute updated *epistemic* prediction covariance
        COV = self.BetaInv + np.einsum("nki,ij,nlj->nkl", G, Ainv, G)

        # pull out standard deviations
        get_diag = vmap(jnp.diag, (0,))
        stdvs = np.sqrt(get_diag(COV))

        return stdvs

    # function to predict from posterior samples
    def predict_MCMC(self, X):
        # make point predictions
        preds = jit(vmap(lambda params: self.forward_batch(params, X), (0,)))(self.posterior_params)

        # take mean and standard deviation
        stdvs = np.sqrt(np.diag(self.BetaInv) + np.var(preds, 0))
        preds = np.mean(preds, 0)

        return preds, stdvs

    # return indeces of optimal samples
    def searchEI(self, data, objective, N, batch_size = 512):

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # make predictions once
        all_preds  = []
        for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
            # make predictions on data
            all_preds.append(self.forward_batch(self.params, data[batch_inds]))

        # compute objective (f: R^[n_t, n_o, w_exp] -> R) in batches
        objective_batch = jit(vmap(lambda pred, stdv: objective(pred, stdv), (0,0)))

        # initialize conditioned parameter covariance
        Ainv_q = jnp.copy(self.Ainv)

        # search for new experiments until find N
        best_experiments = []
        while len(best_experiments) < N:

            # compute utilities in batches to avoid memory problems
            utilities = []
            for preds, batch_inds in zip(all_preds, np.array_split(np.arange(n_samples), n_samples//batch_size)):
                stdvs = self.conditioned_stdv(data[batch_inds], Ainv_q)
                utilities.append(objective_batch(preds, stdvs))
            utilities = jnp.concatenate(utilities)
            print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            plt.plot(np.array(utilities).ravel())

            # pick an experiment
            print(f"Picked experiment {len(best_experiments)} out of {N}")
            exp = np.argmax(utilities)

            # add experiment to the list
            best_experiments += [exp.item()]

            # compute sensitivity to sample
            Gi = self.Gi(self.params, data[exp])

            # update conditioned parameter covariance
            Ainv_q -= self.Ainv_next(Gi, Ainv_q, self.BetaInv)

        plt.show()
        return best_experiments


    # compute utility of each experiment
    def fast_utility(self, predCOV):
        # return log det of prediction covariance
        # predCOV has shape [n_out, n_out]
        # log eig predCOV has shape [n_out]
        # return scalar
        return jnp.nansum(jnp.log(jnp.linalg.eigvalsh(predCOV)))

    # return indeces of optimal samples
    def search(self, data, objective, N, Ainv_q = None,
               min_explore = 1e-4, max_explore = 1e4, batch_size=512):

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # compute profit function (f: R^[n_t, n_o] -> R) in batches
        objective_batch = jit(vmap(lambda pred: objective(pred)))

        f_P = []
        for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
            # make predictions on data
            preds = self.predict_point(data[batch_inds])
            f_P.append(objective_batch(preds))
        f_P = jnp.concatenate(f_P).ravel()
        print("Top 5 profit predictions: ", jnp.sort(f_P)[::-1][:5])

        # if explore <= 0, return pure exploitation search
        if min_explore <= 0.:
            print("Pure exploitation, returning N max objective experiments")
            return np.array(jnp.argsort(f_P)[::-1][:N])

        # initialize with sample that maximizes objective
        best_experiments = [np.argmax(f_P).item()]
        print(f"Picked experiment {len(best_experiments)} out of {N}")

        # init and update parameter covariance
        if Ainv_q is None:
            Ainv_q = jnp.copy(self.Ainv)
        Gi = self.Gi(self.params, data[best_experiments[-1]])
        Ainv_q -= self.Ainv_next(Gi, Ainv_q, self.BetaInv)

        # define batch function to compute utilities over all samples
        utility_batch = jit(vmap(self.fast_utility))

        # search for new experiments until find N
        while len(best_experiments) < N:

            # compute information acquisition function
            f_I = []
            for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
                # predCOV has shape [n_samples, n_out, n_out]
                predCOV = self.compute_searchCOV(self.Beta, self.G(self.params, data[batch_inds]), Ainv_q)
                f_I.append(utility_batch(predCOV))
            f_I = jnp.concatenate(f_I).ravel()

            # select next point
            w = min_explore
            while jnp.argmax(f_P + w*f_I) in best_experiments and w < max_explore:
                w += min_explore
            utilities = f_P + w*f_I
            print("Exploration weight set to: {:.4f}".format(w))
            print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(utilities)[::-1]
            for exp in exp_sorted:
                # accept experiment if unique
                if exp not in best_experiments:
                    best_experiments += [exp.item()]

                    # update parameter covariance given selected condition
                    Gi = self.Gi(self.params, data[best_experiments[-1]])
                    Ainv_q -= self.Ainv_next(Gi, Ainv_q, self.BetaInv)
                    print(f"Picked experiment {len(best_experiments)} out of {N}")

                    # if have enough selected experiments, return
                    if len(best_experiments) == N:
                        return best_experiments, Ainv_q
                    else:
                        break
