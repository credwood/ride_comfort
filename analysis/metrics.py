
import numpy as np

def compute_nmv(Ax_95, Ay_95, Az_95):
    return 6 * np.sqrt(Ax_95**2 + Ay_95**2 + Az_95**2)

def compute_nvd(Axp_50, Ayp_50, Azp_50, Ayp_95):
    return 3 * np.sqrt(16 * Axp_50**2 + 4 * Ayp_50**2 + Azp_50**2) + 5 * Ayp_95

def compute_nva(Azp_95, Aya_95, Aza_95, Axd_95):
    return 4 * (Azp_95) + 2 * np.sqrt(Aya_95**2 + Aza_95**2) + 4 * Axd_95

def compute_av(X, Y, Z):
    return np.sqrt(X**2 + Y**2 + Z**2)

def compute_vdv(signal):
    return np.sum(signal**4) ** 0.25

def compute_vdv_ratios(VDV, Aw):
    return VDV / (Aw + 1e-12)
