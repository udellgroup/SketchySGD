## Deep Learning Experiments

By default, the deep learning experiments will log to Weights and Biases. Logging to Weights and Biases is required to run the entire training pipeline.
Please use [https://github.com/facebookresearch/optimizers/tree/main](https://github.com/facebookresearch/optimizers/tree/main) to install DistributedShampoo, which is one of the optimizers we compare to in our experiments.

To reproduce our experiments, please do the following:

1. Download the required datasets to a new folder `data` by running `python get_data.py`. This will download and split the data into train, validation, and test sets.
2. Next, perform hyperparameter tuning by running all scripts of the form `config/tune_*.sh`. This will perform a grid search over the learning rate for each optimizer and dataset.
3. Now, obtain final results by running all scripts of the form `config/final_*.sh`. This will train the models with the best learning rate (found in the previous step) over several random seeds.
4. Once the final results have been obtained, run the notebook `plotting/performance_plots.ipynb` to generate the plots in the paper.