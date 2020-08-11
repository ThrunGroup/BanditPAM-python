#!/bin/bash
#SBATCH --partition=sc-quick --qos=normal
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=107G

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
cd /sailhome/motiwari/manualPAM
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.KMedoidsFastPAM1 -kmeans.k 5 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.KMedoidsFastPAM -kmeans.k 5 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.CLARA -kmeans.k 5 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.FastCLARA -kmeans.k 5 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.CLARANS -kmeans.k 5 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.FastCLARANS -kmeans.k 5 -resulthandler DiscardResultHandler

java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.KMedoidsFastPAM1 -kmeans.k 10 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.KMedoidsFastPAM -kmeans.k 10 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.CLARA -kmeans.k 10 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.FastCLARA -kmeans.k 10 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.CLARANS -kmeans.k 10 -resulthandler DiscardResultHandler
java -jar elki-bundle-0.7.5.jar KDDCLIApplication -verbose -verbose -dbc.in /sailhome/motiwari/manualPAM/MNIST-65k-truncated.csv -algorithm clustering.kmeans.FastCLARANS -kmeans.k 10 -resulthandler DiscardResultHandler

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"
