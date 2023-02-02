# SketchySGD_ICML

Code for ICML 2023 submission.

To run comparisons between SketchySGD and other optimizers:
1. Run download_data.py to get datasets from libsvm
2. Run config/logistic.sh to run logistic regression experiments + run config/least_squares.sh to run ridge regression experiments
3. Run the jupyter notebook plot_general_results.ipynb to generate plots

To run ablation experiments for SketchySGD:
1. Run download_data.py to get datasets from libsvm
2. Run config/ablation.sh to run ablation experiments
3. Run the jupyter notebook plot_ablation_results.ipynb to generate plots
