#!/bin/bash
#SBATCH --job-name=id_diffusion
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00

# ── Environment ──────────────────────────────────────────────────────────────
module purge
# Check and load Python module - adjust version if needed
module load python/3.10 2>/dev/null || module load python/3.9 2>/dev/null || module load python 2>/dev/null || echo "Warning: No Python module loaded"
# Check and load CUDA module - adjust version if needed
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: No CUDA module loaded"

# Try to load conda/miniconda if available
module load miniconda 2>/dev/null || module load anaconda 2>/dev/null || module load conda 2>/dev/null || echo "Warning: No conda module loaded"

# Activate your conda/venv environment — adjust the path if needed
source ~/.env 2>/dev/null || true
conda create -n "cv_diffusion" python=3.10 -y 2>/dev/null || true
conda activate cv_diffusion 2>/dev/null || source /path/to/venv/bin/activate 2>/dev/null || echo "Warning: Could not activate conda environment"

# Install requirements if not already installed
pip install -r requirements.txt --quiet || echo "Warning: Failed to install requirements"

# Move into the project directory
PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR" || exit 1

# Create output / log directories if they don't exist
mkdir -p logs output

echo "========================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Working dir : $(pwd)"
echo "Started     : $(date)"
echo "========================================"

# ── Optional: cache HuggingFace models on scratch so they survive across jobs
export HF_HOME="hf_cache"
mkdir -p "$HF_HOME"

# ── Run ──────────────────────────────────────────────────────────────────────
python main.py

EXIT_CODE=$?
echo "========================================"
echo "Finished : $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"
exit $EXIT_CODE
