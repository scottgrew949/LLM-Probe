"""experiments/t1d_gpt2/run_experiment.py — T1d on GPT-2 medium."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.t1_experiments import T1dExperiment

T1dExperiment("t1d_gpt2").run()
