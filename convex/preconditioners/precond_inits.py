from preconditioners.nystrom import Nystrom
from preconditioners.ssn import SSN

def init_preconditioner(precond_type, model, rho, rank):
    if precond_type == 'nystrom':
        return Nystrom(model, rho, rank)
    elif precond_type == 'ssn':
        return SSN(model, rho)
    else:
        raise ValueError(f"We do not support the following preconditioner type: {precond_type}")