import numpy as np
from scipy.sparse.linalg import LinearOperator

def appx_newton_step(U, S, rho, g):
    UTg = U.T @ g
    vns = (U @ np.divide(UTg, S + rho) + (g - U @ UTg)/rho)
    return vns

def appx_newton_step2(U, S_mod, rho, g):
    g_mod = g / rho
    UUTg_mod = U @ (U.T @ g_mod)
    return g_mod - S_mod * UUTg_mod

def get_lin_op(D, A, U, S, rho):
    def mv1(x):
        return U @ (np.reciprocal(np.sqrt(S + rho)) * (U.T @ x)) + 1 / np.sqrt(rho) * (x - U @ (U.T @ x))
    def mv2(x):
        return A @ x
    def mv3(x):
        return np.sqrt(D) * x

    def mv2_t(x):
        return A.T @ x

    return LinearOperator((D.shape[0], U.shape[0]),
                          matvec = lambda x: mv3(mv2(mv1(x.reshape(-1)))),
                          rmatvec = lambda x: mv1(mv2_t(mv3(x.reshape(-1)))))

def get_lin_op2(D, A, U, S):
    def mv(x):
        return A.T @ (D * (A @ x)) - U @ (S * (U.T @ x))

    return LinearOperator((U.shape[0], U.shape[0]), matvec = mv, rmatvec = mv)
