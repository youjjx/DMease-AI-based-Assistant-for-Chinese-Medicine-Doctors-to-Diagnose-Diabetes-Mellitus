"""Measured compound affinity research helpers; never treat scores as efficacy."""
import hashlib
import math

import numpy as np


def pk_from_nm(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Affinity must be a finite positive concentration")
    return 9.0 - math.log10(value)


def standardized_structure(smiles: str) -> tuple[str, str]:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid molecular structure")
    parent = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(mol))
    Chem.SanitizeMol(parent)
    canonical = Chem.MolToSmiles(parent, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False) or "ACYCLIC"
    return canonical, scaffold


def scaffold_split(scaffold: str) -> str:
    bucket = int(hashlib.sha256(("dmease-affinity-v3|" + scaffold).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def fingerprints(smiles: list[str], radius: int = 2, bits: int = 2048) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits, includeChirality=True)
    return np.asarray([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def regression_metrics(gold, predicted):
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    gold, predicted = np.asarray(gold), np.asarray(predicted)
    rho = float(spearmanr(gold, predicted).statistic) if np.ptp(gold) > 1e-12 and np.ptp(predicted) > 1e-12 else None
    return {"records": len(gold), "MAE_pK": float(mean_absolute_error(gold, predicted)),
            "RMSE_pK": float(math.sqrt(mean_squared_error(gold, predicted))),
            "R2": float(r2_score(gold, predicted)), "Spearman": rho if rho is None or math.isfinite(rho) else None}


def scaffold_bootstrap(gold, selected, baseline, groups, repeats=2000, seed=42):
    """Paired cluster bootstrap: preserve within-scaffold dependence."""
    delta = np.abs(np.asarray(gold)-selected) - np.abs(np.asarray(gold)-baseline)
    unique = sorted(set(groups))
    indices = [np.flatnonzero(np.asarray(groups) == name) for name in unique]
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(delta[np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])])) for _ in range(repeats)]
    return {"MAE_selected_minus_mean": float(np.mean(delta)), "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "scaffolds": len(unique), "replicates": repeats, "seed": seed,
            "interpretation": "Descriptive paired scaffold-bootstrap interval; no multiple-comparison correction or clinical claim"}
