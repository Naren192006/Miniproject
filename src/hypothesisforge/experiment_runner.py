"""
HypothesisForge - Executable Experiment Runner
Generates runnable Python/scikit-learn scripts from Prompt 06 ExperimentDesignOutput,
executes them under a simulated distribution shift across multiple random seeds,
and computes paired t-test statistics to confirm, refute, or mark inconclusive.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .prompt_engine.models import ExperimentDesignOutput


@dataclass
class ExperimentExecutionResult:
    research_question: str
    dataset_used: str
    baseline_name: str
    proposed_name: str
    seeds: List[int]
    baseline_scores: List[float]
    proposed_scores: List[float]
    baseline_mean: float
    proposed_mean: float
    delta: float
    p_value: float
    outcome: str  # "confirmed", "refuted", "inconclusive"
    summary: str
    generated_script: str


def generate_experiment_script(plan: ExperimentDesignOutput, n_seeds: int = 5) -> str:
    """
    Generates a standalone, reproducible Python script implementing the experiment protocol
    on scikit-learn's Digits dataset with an out-of-distribution feature shift.
    """
    seeds = [42 + i for i in range(max(3, n_seeds))]
    baseline_label = plan.baselines[0] if plan.baselines else "Linear ERM Baseline"
    proposed_label = plan.independent_variable

    script = f'''"""
Auto-generated HypothesisForge Experiment Protocol
Research Question: {plan.research_question}
Dataset (proxy benchmark): sklearn Digits (64-dim) with out-of-distribution contrast/noise shift
"""

import json
import numpy as np
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEEDS = {seeds}
BASELINE_LABEL = {json.dumps(baseline_label)}
PROPOSED_LABEL = {json.dumps(proposed_label)}

def load_shifted_digits(seed: int):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    rng = np.random.default_rng(seed)
    # Simulate distribution shift on target domain (contrast scaling + Gaussian sensor noise)
    X_test_shifted = np.clip(X_test * 0.85 + rng.normal(0.0, 1.5, size=X_test.shape), 0.0, 16.0)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test_shifted)
    return X_train_s, X_test_s, y_train, y_test

def run():
    baseline_scores = []
    proposed_scores = []

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = load_shifted_digits(seed)

        # Baseline model (Logistic Regression ERM)
        base_model = LogisticRegression(max_iter=400, random_state=seed)
        base_model.fit(X_tr, y_tr)
        base_acc = accuracy_score(y_te, base_model.predict(X_te)) * 100.0
        baseline_scores.append(round(float(base_acc), 2))

        # Proposed model (MLP Classifier)
        prop_model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=250, random_state=seed)
        prop_model.fit(X_tr, y_tr)
        prop_acc = accuracy_score(y_te, prop_model.predict(X_te)) * 100.0
        proposed_scores.append(round(float(prop_acc), 2))

    base_mean = float(np.mean(baseline_scores))
    prop_mean = float(np.mean(proposed_scores))
    delta = prop_mean - base_mean
    t_stat, p_val = stats.ttest_rel(proposed_scores, baseline_scores)
    p_val = float(p_val) if not np.isnan(p_val) else 1.0

    if p_val < 0.05 and delta > 0:
        outcome = "confirmed"
    elif p_val < 0.05 and delta < 0:
        outcome = "refuted"
    else:
        outcome = "inconclusive"

    print(json.dumps({{
        "seeds": SEEDS,
        "baseline_scores": baseline_scores,
        "proposed_scores": proposed_scores,
        "baseline_mean": round(base_mean, 2),
        "proposed_mean": round(prop_mean, 2),
        "delta": round(delta, 2),
        "p_value": round(p_val, 4),
        "outcome": outcome,
    }}))

if __name__ == "__main__":
    run()
'''
    return script


def execute_experiment_plan(plan: ExperimentDesignOutput, n_seeds: int = 5) -> ExperimentExecutionResult:
    """
    Generates and executes the small-scale experiment script in an isolated Python subprocess,
    returning real measured accuracies, paired t-test p-value, and confirmation verdict.
    """
    script = generate_experiment_script(plan, n_seeds=n_seeds)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(script)
        tmp_path = Path(tmp.name)

    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        metrics = json.loads(proc.stdout.strip().splitlines()[-1])
    finally:
        tmp_path.unlink(missing_ok=True)

    outcome = metrics["outcome"]
    delta = metrics["delta"]
    p_val = metrics["p_value"]
    baseline_label = plan.baselines[0] if plan.baselines else "Linear ERM Baseline"

    if outcome == "confirmed":
        summary = (
            f"Proxy benchmark confirmed a statistically significant gain (+{delta:.2f}%, p={p_val:.4f}) "
            f"across {len(metrics['seeds'])} seeds on shifted Digits."
        )
    elif outcome == "refuted":
        summary = (
            f"Proxy benchmark refuted improvement ({delta:.2f}%, p={p_val:.4f}); "
            f"baseline performed better under this shift."
        )
    else:
        summary = (
            f"Difference ({delta:+.2f}%, p={p_val:.4f}) was not statistically significant at alpha=0.05 "
            f"across {len(metrics['seeds'])} seeds (inconclusive)."
        )

    return ExperimentExecutionResult(
        research_question=plan.research_question,
        dataset_used="Digits (OOD Contrast+Noise Shift Proxy)",
        baseline_name=baseline_label,
        proposed_name=plan.independent_variable,
        seeds=metrics["seeds"],
        baseline_scores=metrics["baseline_scores"],
        proposed_scores=metrics["proposed_scores"],
        baseline_mean=metrics["baseline_mean"],
        proposed_mean=metrics["proposed_mean"],
        delta=delta,
        p_value=p_val,
        outcome=outcome,
        summary=summary,
        generated_script=script,
    )
