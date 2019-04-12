experiment=C
data_folder="/cluster/home/sanagnos/NLU/data/"
save_model_folder="/cluster/home/sanagnos/NLU/models/"
save_results_folder="/cluster/home/sanagnos/NLU/results/"

module load python_gpu/3.6.4
python -m pip install --user gensim
bsub -n 2 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python src/run_experiment.py --experiment ${experiment} \
    --data_folder=${data_folder} --save_model_folder=${save_model_folder} --save_results_folder=${save_results_folder}
