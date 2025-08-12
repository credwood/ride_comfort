from typing import Tuple

import numpy as np
import pandas as pd

def compute_nmv(Ax_95: float, Ay_95: float, Az_95: float) -> float:
    return 6 * np.sqrt(Ax_95**2 + Ay_95**2 + Az_95**2)

def compute_nvd(Axp_50: float, Ayp_50: float, Azp_50: float, Ayp_95: float) -> float:
    return 3 * np.sqrt(16 * Axp_50**2 + 4 * Ayp_50**2 + Azp_50**2) + 5 * Ayp_95

def compute_nva(Azp_95: float, Aya_95: float, Aza_95: float, Axd_95: float) -> float:
    return 4 * (Azp_95) + 2 * np.sqrt(Aya_95**2 + Aza_95**2) + 4 * Axd_95

def compute_av(X: float, Y: float, Z: float) -> float:
    return np.sqrt(X**2 + Y**2 + Z**2)

def compute_vdv(signal: pd.Series) -> float:
    return np.sum(signal**4) ** 0.25

def compute_vdv_ratios(VDV: float, Aw: float):
    return VDV / (Aw + 1e-12)

def get_mean_and_std(signal: pd.Series) -> Tuple[float, float]:
    return np.mean(signal).item(), np.std(signal).item()