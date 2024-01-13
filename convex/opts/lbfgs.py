import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time

class LBFGS():
    """Wrapper around scipy.optimize.fmin_l_bfgs_b
    """
    def __init__(self,
                model,
                Mem,
                factr):
        """Initialize L-BFGS

        Args:
            model (Logistic/LeastSquares): Object containing problem information
            Mem (int): Memory size
            factr: Tolerance for convergence
        """
        self.model = model
        self.Mem = Mem
        self.factr = factr
        self.loss_ls = lambda x: self.model.get_losses(v=x, both=False)['train_loss']
        self.grad_ls = lambda x: self.model.get_grad(np.arange(self.model.ntr), x)

    def run(self, max_iters):
        iteration_times = []
        function_values = []
        iterates = []
        last_time = [time.time()]  # Use a list to make it mutable inside the callback

        # Callback function to store the function value and time taken at each iteration
        def callback(xk):
            current_time = time.time()
            iteration_times.append(current_time - last_time[0])
            last_time[0] = current_time

            # Store the current iterate
            iterates.append(xk)
            
            # Evaluate and store the function value at the current parameters
            function_values.append(self.loss_ls(xk))

        # time = timeit.default_timer()
        fmin_l_bfgs_b(func=self.loss_ls,
                                x0=self.model.w,
                                fprime=self.grad_ls,
                                m=self.Mem,
                                factr=self.factr,
                                maxiter=max_iters,
                                iprint=1,
                                callback=callback)
        # print(f"Time taken: {timeit.default_timer() - time}")
        for i, (t, f_val) in enumerate(zip(iteration_times, function_values), 1):
            print(f"Iteration {i}: Time = {t:.6f} seconds, Function Value = {f_val:.6f}")

        # Print the total time taken
        print(f"Total time taken: {sum(iteration_times):.6f} seconds")

        return {'times': iteration_times,
                 'train_loss': function_values,
                 'iterates': iterates}