import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_comfort_thresholds_nvm(nmvs, categories):
    plt.figure()

    for triax, nmv in nmvs:
        plt.scatter(1, nmv, label=f'triax: {triax} N_M_V')

    plt.xlim(0, 2)
    plt.ylim(0, 5)

    for (lo, hi), label in categories:
        plt.axhline(lo, linestyle='-', color='gray', linewidth=0.5)
        plt.text(1.5, lo + 0.1, label, fontsize=8, verticalalignment='bottom')

    plt.legend(loc='lower left')
    plt.title('Mean Comfort (Standard Method) Index NMV')
    plt.xticks([])
    plt.gca()
    plt.tight_layout()
    plt.show()

def plot_comfort_thresholds_nvd_nva(nvds, nvaz, categories, pair_dict):
    plt.figure()

    for triax, nvd in nvds:
        if nvd is None:
            continue
        plt.scatter(1, nvd, label=f'triax: {triax} Standing (NVD)', marker='x')
    
    for triax, nva in nvaz:
        if nva is None:
            continue
        orientation = [label for (_, s), label in pair_dict if s == triax][0]
        plt.scatter(1, nva, label=f'triax: {triax} Seated (NVA), {orientation}', marker='o')
    
    plt.xlim(0, 2)
    plt.ylim(0, 5)

    for (lo, hi), label in categories:
        plt.axhline(lo, linestyle='-', color='gray', linewidth=0.5)
        plt.text(1.5, lo + 0.1, label, fontsize=8, verticalalignment='bottom')

    plt.legend(loc='upper left')
    plt.title('Mean Comfort (Complete Method) Index')
    plt.xticks([])
    plt.gca()
    plt.tight_layout()
    plt.show()

def plot_vdv_over_time(t, vdv_x, vdv_y, vdv_z):
    t = [i for i in range(len(t))]

    plt.figure()
    plt.plot(t, vdv_x, label='VDV_X')
    plt.plot(t, vdv_y, label='VDV_Y')
    plt.plot(t, vdv_z, label='VDV_Z')
    plt.ylabel('Vibration Dose Value (m/s^1.75)')
    plt.xlabel('Time (seconds elapsed)')
    plt.title('Vibration Dose Over Time (1s interval)')
    plt.legend(loc='upper right')
    plt.gca()
    plt.tight_layout()
    plt.show()

def plot_ratio_comparison(t, ratios, labels):
    t = [i for i in range(len(t))]

    plt.figure()
    for ratio, label in zip(ratios, labels):
        plt.plot(t, ratio, label=label)
    plt.axhline(1.75, linestyle='-', color='gray', label='VDV / (a_wT^1/4) = 1.75')
    plt.ylim(0, 2)
    plt.ylabel('VDV / (a_wT^1/4)')
    plt.xlabel('Time (seconds elapsed)')
    plt.title('VDV method compared to basic method')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_comfort_timeseries(t_5s_minutes, Cx, Cy, Cz):
    
    t_5s_minutes = [i * 5 for i in range(len(t_5s_minutes))]
    plt.figure()

    # Plot signals
    plt.plot(t_5s_minutes, Cx, marker='o', label='Cx (X)', linestyle='-')
    plt.plot(t_5s_minutes, Cy, marker='s', label='Cy (Y)', linestyle='-', color='orange')
    plt.plot(t_5s_minutes, Cz, marker='^', label='Cz (Z)', linestyle='-', color='green')

    # Comfort bands - these are already defined in the .ipynb

    cats_cont = [
    ((0,0.2), 'Very Comfortable'),
    ((0.2,0.3), 'Comfortable'),
    ((0.3,0.4), 'Medium'),
    ((0.4,1), 'Less Comfortable')
]

    for (lo, hi), label in cats_cont:
        plt.axhspan(lo, hi, color='gray', alpha=0.1)
        y_mid = (lo + hi) / 2
        x_mid = t_5s_minutes[len(t_5s_minutes) // 2]
        plt.text(x_mid, y_mid, label, ha='center', va='center', fontsize=8, color='black', alpha=0.8)

    # Formatting
    plt.ylabel('5s Weighted RMS Acceleration (m/s²)')
    plt.xlabel('Time (seconds elapsed)')
    plt.title('Ride Comfort Over Time (5s Interval)')

    # Limit xticks for readability
    if len(t_5s_minutes) > 15:
        step = len(t_5s_minutes) // 10
        plt.xticks(t_5s_minutes[::step])

    

    plt.legend(loc='upper left')
    plt.gca()
    plt.tight_layout()
    plt.show()


def plot_compare_all_metrics(t_5s_minutes, X_ISO_5s, Y_ISO_5s, Z_ISO_5s, C_cx, C_cy, C_cz):
    """
    Compare ISO and EN comfort metrics at 5s intervals.
    
    Parameters:
        t_5s_minutes: Time vector in minutes (downsampled to 5s intervals)
        X_ISO_5s, Y_ISO_5s, Z_ISO_5s: ISO-weighted rms values downsampled to 5s
        C_cx, C_cy, C_cz: EN comfort components downsampled to 5s
    """
    # Optional: composite EN metric, not plotted but could be
    # C_c = np.sqrt(C_cx**2 + C_cy**2 + C_cz**2)

    t_5s_minutes = [i * 5 for i in range(len(t_5s_minutes))]

    plt.figure(figsize=(10, 6))
    
    # ISO metrics: large filled squares
    plt.scatter(t_5s_minutes, X_ISO_5s, s=125, marker='s', label='ISO:2631 a_W_d_x')
    plt.scatter(t_5s_minutes, Y_ISO_5s, s=125, marker='s', label='ISO:2631 a_W_d_y')
    plt.scatter(t_5s_minutes, Z_ISO_5s, s=125, marker='s', label='ISO:2631 a_W_k_z')

    # EN metrics: filled circles with white edge
    plt.scatter(t_5s_minutes, C_cx, s=40, label='EN:12299 C_c_x', edgecolors='white')
    plt.scatter(t_5s_minutes, C_cy, s=40, label='EN:12299 C_c_y', edgecolors='white')
    plt.scatter(t_5s_minutes, C_cz, s=40, label='EN:12299 C_c_z', edgecolors='white')

    plt.ylabel('Weighted RMS acceleration (m/s²)')
    plt.xlabel('Time (seconds elapsed)')
    plt.title('Compare Ride Comfort Metrics (5s interval)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Downsample x-ticks for clarity
    num_ticks = 5  # adjust as needed
    step = max(1, len(t_5s_minutes) // num_ticks)
    plt.xticks(t_5s_minutes[::step])
    plt.gca()
    plt.tight_layout()
    plt.show()


def plot_iso_timeseries(t, x_iso, y_iso, z_iso, a_v, a_v_5s=None, t_5s=None):
    # Ensure t is float
    if hasattr(t, "dt"):
        t = t.dt.total_seconds() / 60

    t = [i for i in range(len(t))]
    plt.figure()
    plt.plot(t, x_iso, label='a_Wd_x')
    plt.plot(t, y_iso, label='a_Wd_y')
    plt.plot(t, z_iso, label='a_Wk_z')
    plt.plot(t, a_v, linestyle=':', color='black', label='ISO 3D Composite')

    plt.ylabel('1s weighted rms acceleration (m/s²)')
    plt.xlabel('Time (seconds elapsed)')
    plt.title('ISO Ride Comfort (1s Interval)')

    # Downsample x-ticks for clarity
    num_ticks = 5  # adjust as needed
    step = max(1, len(t) // num_ticks)
    plt.xticks(t[::step])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca()
    plt.tight_layout()
    plt.show()


def plot_distributions(C_cx, C_cy, C_cz, triax):
    comfort_data = [
        ("Cx", C_cx, "skyblue"),
        ("Cy", C_cy, "lightgreen"),
        ("Cz", C_cz, "salmon")
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    for ax, (label, data, color) in zip(axes, comfort_data):
        sns.histplot(data, kde=True, ax=ax, bins=20, color=color)
        ax.set_title(f'Distribution of {label} for Triax {triax}', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_xlabel('Comfort Metric Value (m/s²)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_cumulative_distribution(C_cx, C_cy, C_cz, triax):
    comfort_data = [
        ("Cx", C_cx, "skyblue"),
        ("Cy", C_cy, "lightgreen"),
        ("Cz", C_cz, "salmon")
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    for ax, (label, data, color) in zip(axes, comfort_data):
        sns.ecdfplot(data, ax=ax, color=color)
        ax.set_title(f'Cumulative Distribution of {label} for Triax {triax}', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_xlabel('Comfort Metric Value (m/s²)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()  