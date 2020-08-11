#!/bin/bash
#SBATCH --partition=sc-quick --qos=normal
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=107G
#SBATCH --nodelist=scq2

#SBATCH --job-name="sample"
#SBATCH --output=sample-%j.out

# only use the following if you want email notification
#SBATCH --mail-user=motiwari@stanford.edu
#SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
python -u run_profiles.py -e auto_exp_config.py -p

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"
