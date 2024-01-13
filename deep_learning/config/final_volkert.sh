export CUDA_VISIBLE_DEVICES=0

id=41166
data_folder=./data
t_seed=0
entity_name=sketchy-opts
metric=valid_acc
criteria=final
opts=(sgd adam adahessian yogi shampoo sketchysgd)
epochs=105
n_trials=10

dataset=volkert

# Loop over opts
for opt in "${opts[@]}"
do
    python final_run.py --id $id \
                        --data_folder $data_folder \
                        --t_seed $t_seed \
                        --proj_name simods_final_${dataset}_${opt} \
                        --entity_name $entity_name \
                        --tuning_name simods_tune_${dataset}_${opt} \
                        --metric $metric \
                        --criteria $criteria \
                        --opt $opt \
                        --epochs $epochs \
                        --n_trials $n_trials
done

