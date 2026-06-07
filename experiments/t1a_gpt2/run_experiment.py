"""experiments/t1a_gpt2/run_experiment.py — T1a on GPT-2 medium."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.t1_experiments import T1aExperiment

summary = T1aExperiment("t1a_gpt2").run()
sys.exit(0 if summary.get("level3_confirmed") else 1)
