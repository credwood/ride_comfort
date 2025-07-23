
import numpy as np

def compute_nmv(Ax_95, Ay_95, Az_95):
    return 6 * np.sqrt(Ax_95**2 + Ay_95**2 + Az_95**2)

def compute_nvd(Ax_50, Ay_50, Az_50, Ay_95):
    return 3 * np.sqrt(16 * Ax_50**2 + 4 * Ay_50**2 + Az_50**2) + 5 * Ay_95

def compute_av(X, Y, Z):
    return np.sqrt(X**2 + Y**2 + Z**2)

def compute_vdv(signal):
    return np.sum(signal**4) ** 0.25

def compute_vdv_ratios(VDV, Aw):
    return VDV / (Aw + 1e-12)
