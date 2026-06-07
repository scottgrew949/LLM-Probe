"""experiments/t1b_pythia/run_experiment.py — T1b on Pythia 1.4B."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.t1_experiments import T1bExperiment

T1bExperiment("t1b_pythia").run()
