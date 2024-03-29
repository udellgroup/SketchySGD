import random 
from utils import *
import timeit
from models.least_squares import LeastSquares
from models.logistic import Logistic
from opts.sgd import SGD
from opts.svrg import SVRG
from opts.saga import SAGA
from opts.lkatyusha import LKatyusha
from opts.sketchysgd import SketchySGD
from opts.slbfgs import SLBFGS
from opts.lbfgs import LBFGS
from opts.rsn import RSN
from opts.nsketch import NSketch
from opts.nystrom_pcg import NystromPCG
from opts.sparse_sp_pcg import SparseSPPCG
from opts.gauss_sp_pcg import GaussSPPCG
from opts.jacobi_pcg import JacobiPCG

class Experiment():
    def __init__(self, 
                data, 
                model_type, 
                model_params, 
                opt_name, 
                opt_params,
                tol = None,
                min_loss = None,
                time_budget = None,
                verbose = False):
        self.data = data
        self.verbose = verbose

        # Instantiate the model
        self.model_type = model_type
        self.model_params = model_params
        self.model = self.create_model()

        # Instantiate the optimizer
        self.opt_name = opt_name
        self.opt_params = opt_params

        # Set the stopping criteria
        if (tol is not None and min_loss is None) or (tol is None and min_loss is not None):
            raise RuntimeError("You must specify both a tolerance and a minimum loss, or specify neither.")
        if tol is not None and time_budget is not None:
            raise RuntimeError("You must specify either a tolerance or a time budget, not both.")
        self.check_tol = tol is not None
        self.check_time = time_budget is not None
        self.tol = tol
        self.min_loss = min_loss
        self.time_budget = time_budget

    # Return a model of the desired type with the requested data
    def create_model(self):
        # Get the data
        Atr = self.data['Atr']
        btr = self.data['btr']
        Atst = self.data['Atst']
        btst = self.data['btst']

        # Create the model
        if self.model_type == 'logistic':
            model = Logistic(Atr, btr, Atst, btst, **self.model_params)
        elif self.model_type == 'least_squares':
            model = LeastSquares(Atr, btr, Atst, btst, **self.model_params)
        else:
            raise RuntimeError(f"We do not currently support the following model: {self.model_type}.")
        return model

    # Return an optimizer that we can use to train the model
    def create_opt(self):
        if self.opt_name == 'sgd':
            opt = SGD(self.model, **self.opt_params)
        elif self.opt_name == 'svrg':
            opt = SVRG(self.model, **self.opt_params)
        elif self.opt_name == 'saga':
            opt = SAGA(self.model, **self.opt_params)
        elif self.opt_name == 'lkatyusha':
            opt = LKatyusha(self.model, **self.opt_params)
        elif self.opt_name == 'sketchysgd':
            opt = SketchySGD(self.model, **self.opt_params)
        elif self.opt_name == 'slbfgs':
            opt = SLBFGS(self.model, **self.opt_params)
        elif self.opt_name == 'lbfgs':
            opt = LBFGS(self.model, **self.opt_params)
        elif self.opt_name == 'rsn':
            opt = RSN(self.model, **self.opt_params)
        elif self.opt_name == 'nsketch':
            opt = NSketch(self.model, **self.opt_params)
        elif self.opt_name == 'nystrom_pcg':
            opt = NystromPCG(self.model, **self.opt_params)
        elif self.opt_name == 'sparse_sp_pcg':
            opt = SparseSPPCG(self.model, **self.opt_params)
        elif self.opt_name == 'gauss_sp_pcg':
            opt = GaussSPPCG(self.model, **self.opt_params)
        elif self.opt_name == 'jacobi_pcg':
            opt = JacobiPCG(self.model)
        else:
            raise RuntimeError(f"We do not currently support the following optimizer: {self.opt_name}.")
        return opt

    # If update frequency is given in epochs, convert to minibatches
    def preprocess_opt_params(self, bg):
        n_batches = int(np.ceil(self.model.ntr/bg))
        if 'update_freq' in self.opt_params.keys():
            for freq_type, freq_pair in self.opt_params['update_freq'].items():
                if freq_pair[1] == 'epochs':
                    self.opt_params['update_freq'][freq_type] = freq_pair[0] * n_batches
                elif freq_pair[1] == 'minibatches':
                    self.opt_params['update_freq'][freq_type] = freq_pair[0]
                else:
                    raise RuntimeError(f"We do not currently support the following update frequency type: {freq_pair[1]}.")

    # Run the experiment
    def run(self, max_epochs, bg):
        if self.opt_name in ['lbfgs', 'rsn', 'nsketch',
                             'nystrom_pcg', 'sparse_sp_pcg', 'gauss_sp_pcg', 'jacobi_pcg']:
            raise ValueError(f"Please use run_full_grad() for {self.opt_name}.")

        self.preprocess_opt_params(bg)
        opt = self.create_opt()
        if hasattr(opt, 'update_freq'): print(opt.update_freq)

        results = {'times': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'eta': []}

        # Training loop
        for i in range(max_epochs):
            epoch_start = timeit.default_timer()

            batches = minibatch_indices(self.model.ntr, bg)
            # Loop through every minibatch
            for batch in batches:
                opt.step(batch)
            
            epoch_end = timeit.default_timer()
            results['times'].append(epoch_end - epoch_start)

            # Get the results so far
            losses = self.model.get_losses()
            results['train_loss'].append(losses['train_loss'])
            results['test_loss'].append(losses['test_loss'])

            accs = self.model.get_acc()
            results['train_acc'].append(accs['train_acc'])
            results['test_acc'].append(accs['test_acc'])

            results['eta'].append(opt.eta)

            if self.verbose:
                print(f"Train loss at epoch {i}: {losses['train_loss']}, test loss at epoch {i}: {losses['test_loss']}")
                print(f"Train acc. at epoch {i}: {accs['train_acc']}, test acc. at epoch {i}: {accs['test_acc']}")

            # If tolerance is met, stop training
            if self.check_tol and results['train_loss'][-1] - self.min_loss < self.tol:
                break

            # If time budget is met, stop training
            if self.check_time and sum(results['times']) > self.time_budget:
                break

        return results

    def run_full_grad(self, max_epochs):
        if self.opt_name not in ['lbfgs', 'rsn', 'nsketch', 
                                 'nystrom_pcg', 'sparse_sp_pcg', 'gauss_sp_pcg', 'jacobi_pcg']:
            raise ValueError(f"This function is only supported for L-BFGS/RSN/NSketch/PCG methods, not {self.opt_name}.")
        
        opt = self.create_opt()

        if self.opt_name == 'lbfgs':
            results = opt.run(max_epochs)
            results['test_loss'] = [self.model.get_losses(v=x, both=True)['test_loss'] for x in results['iterates']]
            accs = [self.model.get_acc(v=x) for x in results['iterates']]
            results['train_acc'] = [x['train_acc'] for x in accs]
            results['test_acc'] = [x['test_acc'] for x in accs]

            del results['iterates'] 
        else:
            results = {'times': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
            
            for i in range(max_epochs):
                epoch_start = timeit.default_timer()
                opt.step()

                epoch_end = timeit.default_timer()
                results['times'].append(epoch_end - epoch_start)

                # Get the results so far
                losses = self.model.get_losses()
                results['train_loss'].append(losses['train_loss'])
                results['test_loss'].append(losses['test_loss'])

                accs = self.model.get_acc()
                results['train_acc'].append(accs['train_acc'])
                results['test_acc'].append(accs['test_acc'])

                if self.verbose:
                    print(f"Train loss at epoch {i}: {losses['train_loss']}, test loss at epoch {i}: {losses['test_loss']}")
                    print(f"Train acc. at epoch {i}: {accs['train_acc']}, test acc. at epoch {i}: {accs['test_acc']}")

        return results