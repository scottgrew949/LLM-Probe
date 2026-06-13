"""experiments/t1c_gpt2/run_experiment.py — T1c on GPT-2 medium."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.t1_experiments import T1cExperiment

T1cExperiment("t1c_gpt2").run()
