#! /bin/sh

CFG_NAME="prob_ccn_cancer_nc" RESUME=$RESUME sbatch run_hpo.sh
CFG_NAME="prob_ccn_cancer_ppx" RESUME=$RESUME sbatch run_hpo.sh
CFG_NAME="prob_ccn_cancer_stdp" RESUME=$RESUME sbatch run_hpo.sh

CFG_NAME="prob_ccn_iris_nc" RESUME=$RESUME sbatch run_hpo.sh
CFG_NAME="prob_ccn_iris_ppx" RESUME=$RESUME sbatch run_hpo.sh
CFG_NAME="prob_ccn_iris_stdp" RESUME=$RESUME sbatch run_hpo.sh

CFG_NAME="prob_ccn_digits_nc" RESUME=$RESUME sbatch run_hpo.sh
CFG_NAME="prob_ccn_digits_ppx" RESUME=$RESUME sbatch run_hpo.sh
CFG_NAME="prob_ccn_digits_stdp" RESUME=$RESUME sbatch run_hpo.sh