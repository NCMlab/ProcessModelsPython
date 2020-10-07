#!/bin/bash
#SBATCH --job-name=%s.job\n"%(os.path.join(BaseDir, 'jobs', fileName))
#SBATCH --output=%s.out\n"%(os.path.join(BaseDir, 'out', fileName))
#SBATCH --error=%s.err\n"%(os.path.join(BaseDir, 'out', fileName))
#SBATCH --time=01:00:00\n"
#SBATCH --account=def-steffejr-ab\n"
#SBATCH --mem-per-cpu=512M\n\n"
# Added an array for at least one dimension of simulations
#SBATCH --array=1-9\n"
source ~/ENV/bin/activate\n
python ProcessTools.py $SLURM_ARRAY_TASK_ID
