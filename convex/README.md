## Convex Experiments

To reproduce our experiments, please do the following:

1. Download the required datasets to a new folder `data` by running `python download_data.py`. You may have to use the command `data_fixes.sh ./data` to fix some issues in the higgs and susy datasets.
2. Remove the folders `simods_sensitivity_results`, `simods_performance_results`, `simods_performance_results_dup2`, `simods_streaming_results`, `simods_hessian_results`, and `simods_spectra_results`.
3. To generate the plots in section 5.1, 5.2, and 5.3, please run `config/simods_exp_least_squares.sh`, `config/simods_exp_logistic.sh`, `config/simods_exp_least_squares_dup.sh`, and `config/simods_exp_logistic_dup.sh`. Once these scripts are finished running, run the notebook `plotting/simods_performance_results_plots.ipynb`.
4. To generate the plots in section 5.4, please run `config/simods_streaming_exp_logistic.sh`. Once this script has finished running, run the notebook `plotting/simods_streaming_results_plots.ipynb`.
5. To generate the plots in section 5.4, please run `config/simods_hessian_exp_least_squares.sh` and `config/simods_hessian_exp_logistic.sh`. Once these scripts are finished running, run `plotting/simods_hessian_results_plots.ipynb`.
6. To generate the learning rate ablation plots in the appendix, please run learning_rate_ablation.ipynb. Once this notebook is finished running, run `plotting/learning_rate_ablation_plot.ipynb`.
7. To generate the sensitivity plots in the appendix, please run `config/simods_sensitivity_exp_least_squares.sh`, `config/simods_sensitivity_exp_logistic.sh`, and `config/simods_compute_spectra.sh`. Once these scripts are finished running, run `plotting/simods_sensitivity_results_plots.ipynb` and `plotting/simods_spectrum_results_plots.ipynb`.

Running all the experiments can take a lot of time. If you would like to generate plots based on the existing results in `simods_sensitivity_results`, `simods_performance_results`, `simods_streaming_results`, `simods_hessian_results`, and `simods_spectra_results` just run the corresponding notebooks mentioned above.