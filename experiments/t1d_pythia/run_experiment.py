"""experiments/t1d_pythia/run_experiment.py — T1d on Pythia 1.4B."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.t1_experiments import T1dExperiment

T1dExperiment("t1d_pythia").run()
