import matplotlib.pyplot as plt
import numpy as np

def plot_comfort_thresholds(nmv, nvd, categories):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(1, nmv, label='N_M_V')
    plt.scatter(1, nvd, label='N_V_D')
    plt.xlim(0, 2)
    plt.ylim(0, 5)

    for (lo, hi), label in categories:
        plt.axhline(lo, linestyle='-', color='gray', linewidth=0.5)
        plt.text(1.5, lo + 0.1, label, fontsize=8, verticalalignment='bottom')

    plt.legend(loc='lower left')
    plt.title('Mean Comfort Index')
    plt.xticks([])
    plt.gca().set_aspect(2 / 3)
    plt.tight_layout()
    plt.show()

def plot_vdv_over_time(t, vdv_x, vdv_y, vdv_z):
    plt.figure()
    plt.plot(t, vdv_x, label='VDV_X')
    plt.plot(t, vdv_y, label='VDV_Y')
    plt.plot(t, vdv_z, label='VDV_Z')
    plt.ylabel('Vibration Dose Value (m/s^1.75)')
    plt.title('Vibration Dose Over Time (1s interval)')
    plt.legend(loc='east')
    plt.gca().set_aspect(3 / 2)
    plt.show()

def plot_ratio_comparison(t, ratios, labels):
    plt.figure()
    for ratio, label in zip(ratios, labels):
        plt.plot(t, ratio, label=label)
    plt.axhline(1.75, linestyle='-', color='gray', label='VDV / (a_wT^1/4) = 1.75')
    plt.ylim(0, 2)
    plt.ylabel('VDV / (a_wT^1/4)')
    plt.title('VDV method compared to basic method')
    plt.legend(loc='lower right')
    plt.show()

def plot_comfort_timeseries(t_5s, cx, cy, cz):
    plt.figure()
    plt.scatter(t_5s, cx, label='C_c_x', marker='o')
    plt.scatter(t_5s, cy, label='C_c_y', marker='s')
    plt.scatter(t_5s, cz, label='C_c_z', marker='^')

    for threshold, label in zip([0.0, 0.2, 0.3, 0.4],
                                 ['Very comfortable', 'Comfortable', 'Medium', 'Less comfortable']):
        plt.axhline(threshold, linestyle='-', label=label, linewidth=0.5)

    plt.ylim(0, 0.5)
    plt.ylabel('5s weighted rms acceleration (m/s²)')
    plt.title('EN:12299 Ride Comfort (5s Interval)')
    plt.legend(loc='east')
    plt.gca().set_aspect(3 / 2)
    plt.show()

def plot_iso_timeseries(t, x_iso, y_iso, z_iso, a_v, a_v_5s=None, t_5s=None):
    plt.figure()
    plt.plot(t, x_iso, label='X ISO_WD')
    plt.plot(t, y_iso, label='Y ISO_WD')
    plt.plot(t, z_iso, label='Z ISO_WB')
    plt.plot(t, a_v, linestyle=':', color='black', label='ISO 3D Composite')

    if a_v_5s is not None and t_5s is not None:
        plt.scatter(t_5s, a_v_5s, label='5s Composite', color='red', s=20)

    plt.ylabel('1s weighted rms acceleration (m/s²)')
    plt.title('ISO Ride Comfort (1s Interval)')

    # Optional: downsample ticks for readability
    if len(t) > 15:
        step = len(t) // 15
        plt.xticks(t[::step])

    plt.legend(loc='east')
    plt.gca().set_aspect(3 / 2)
    plt.show()
