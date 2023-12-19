import numpy as np
import jax.numpy as jnp
from jax import nn, jacfwd, jit, vmap, lax, random
from jax.scipy.linalg import block_diag


# jit compile Hessian computation step
@jit
def A_next(G, Beta):
    A = jnp.einsum('nki,kl,nlj->ij', G, Beta, G)
    return (A + A.T) / 2.


# jit compile diagonal Hessian computation step
@jit
def A_next_diag(G, Beta):
    A = jnp.einsum('nki,kl,nli->i', G, Beta, G)
    return (A + A.T) / 2.

@jit
def trace_GGM(G, M):
    # return Trace( g @ g.T @ M)
    # k indexes each output
    return jnp.einsum("nki,nkj,ij->k", G, G, M)

@jit
def Beta_hs(stdvs):
    # if stdvs is shape [n_s, n_out]
    # returns n_s precision matrices, shape [n_s, n_out, n_out]
    return vmap(lambda stdv: jnp.diag(1. / stdv ** 2))(stdvs)


# jit compile diagonal Hessian computation step when precision varies for each sample
@jit
def A_next_dhs(Gm, Gs, stdvs):
    Beta = Beta_hs(stdvs)
    A = jnp.einsum('nki,nkl,nli->i', Gm, Beta, Gm)
    A += jnp.einsum('nki,nkl,nli->i', Gs, Beta, Gs)
    return (A + A.T) / 2.


# jit compile Hessian computation step when precision varies for each sample
@jit
def A_next_hs(Gm, Gs, stdvs):
    Beta = Beta_hs(stdvs)
    A = jnp.einsum('nki,nkl,nlj->ij', Gm, Beta, Gm)
    A += jnp.einsum('nki,nkl,nlj->ij', Gs, Beta, Gs)
    return (A + A.T) / 2.


# jit compile Hessian computation step for mixture model
@jit
def A_next_mix(Gm, Gs, stdvs, mixs):

    # Hessian for component 1
    A  = jnp.einsum('n,ni,n,nj->ij', mixs[:, 0], Gm[:, 0], (1./stdvs[:, 0]**2), Gm[:, 0])
    A += jnp.einsum('n,ni,n,nj->ij', mixs[:, 0], Gs[:, 0], (1./stdvs[:, 0]**2), Gs[:, 0])

    # Hessian for component 2
    A += jnp.einsum('n,ni,n,nj->ij', mixs[:, 1], Gm[:, 1], (1./stdvs[:, 1]**2), Gm[:, 1])
    A += jnp.einsum('n,ni,n,nj->ij', mixs[:, 1], Gs[:, 1], (1./stdvs[:, 1]**2), Gs[:, 1])

    return (A + A.T) / 2.


# jit compile function to compute log of determinant of a matrix
@jit
def log_det(A):
    L = jnp.linalg.cholesky(A)
    return 2 * jnp.sum(jnp.log(jnp.diag(L)))


# jit compile inverse Hessian computation step
@jit
def Ainv_next(G, Ainv, BetaInv):
    GAinv = G @ Ainv
    Ainv_step = GAinv.T @ jnp.linalg.inv(BetaInv + GAinv @ G.T) @ GAinv
    Ainv_step = (Ainv_step + Ainv_step.T) / 2.
    return Ainv_step


# approximate inverse of A, where A = LL^T, Ainv = Linv^T Linv
@jit
def compute_Ainv(A):
    Linv = jnp.linalg.inv(jnp.linalg.cholesky(A))
    Ainv = Linv.T @ Linv
    return Ainv


# jit compile measurement covariance computation
@jit
def compute_yCOV(errors, G, Ainv):
    return jnp.einsum('nk,nl->kl', errors, errors) + jnp.einsum("nki,ij,nlj->kl", G, Ainv, G)


# jit compile measurement covariance computation
@jit
def compute_yCOV_diag(errors, G, Ainv):
    return jnp.einsum('nk,nl->kl', errors, errors) + jnp.einsum("nkj,j,nlj->kl", G, Ainv, G)


# jit compile prediction covariance computation
@jit
def compute_predCOV(BetaInv, G, Ainv):
    return BetaInv + jnp.einsum("nki,ij,nlj->nkl", G, Ainv, G)

# jit compile prediction covariance computation
@jit
def compute_epistemic_COV(G, Ainv):
    return jnp.einsum("nki,ij,nlj->nkl", G, Ainv, G)


# jit compile prediction covariance computation
@jit
def compute_predCOV_diag(BetaInv, G, Ainv):
    return BetaInv + jnp.einsum("nki,i,nli->nkl", G, Ainv, G)


# jit compile prediction covariance computation
@jit
def compute_predCOV_hs(Gm, Gs, Ainv):
    return jnp.einsum("nki,ij,nlj->nkl", Gm, Ainv, Gm) + jnp.einsum("nki,ij,nlj->nkl", Gs, Ainv, Gs)


# function to make sure a matrix is positive definite
# (algorithm 3.3 in Numerical Optimization)
def make_pos_def(A, alpha, beta=1e-5):
    if jnp.min(jnp.diag(A)) > 0:
        tau = 0.
    else:
        tau = beta - jnp.min(jnp.diag(A))

    # increase precision of prior until posterior precision is positive definite
    A += tau * jnp.diag(alpha)
    while jnp.isnan(jnp.linalg.cholesky(A)).any():
        # increase prior precision
        tau = np.max([2 * tau, beta])
        A += tau * jnp.diag(alpha)

    return A
