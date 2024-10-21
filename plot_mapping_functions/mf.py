import numpy as np
import matplotlib.pyplot as plt

def plot_mf(name):

    R = 6371
    H_slm = 450
    slm = lambda z: 1 / (np.cos(np.arcsin(R/(R+H_slm)*np.sin(z))))
    H_opt = 506.7
    alpha = 0.9782
    mslm = lambda z: 1 / (np.cos(np.arcsin(R/(R+H_opt)*np.sin(alpha*z))))
    klobuchar = lambda z: 1.0 + 2 * ((96-(90-(z/np.pi*180)))/90)**3

    fig = plt.figure(figsize=(6, 3))
    angles = np.linspace(0, np.pi/2, 180)
    if name == "SLM":
        plt.plot(angles/np.pi*180, slm(angles), label="SLM-MF")
    if name == "MSLM":
        plt.plot(angles/np.pi*180, mslm(angles), label="MSLM-MF", c="tab:orange")
    if name == "Klobuchar":
        plt.plot(angles/np.pi*180, klobuchar(angles), label="Klobuchar-MF")
    if name == "SLMMSLM":
        plt.plot(angles/np.pi*180, slm(angles), label="SLM-MF")
        plt.plot(angles/np.pi*180, mslm(angles), label="MSLM-MF")
    if name == "all":
        plt.plot(angles/np.pi*180, slm(angles), label="SLM-MF")
        plt.plot(angles/np.pi*180, mslm(angles), label="MSLM-MF")
        plt.plot(angles/np.pi*180, klobuchar(angles), label="Klobuchar-MF")
    plt.xlabel("zenith angle (z')")
    plt.ylabel("mapping value")
    plt.legend(loc='upper left')
    plt.title("Mapping function", fontweight="bold")
    plt.ylim(0.9, 3.52)
    plt.tight_layout()

    if name == "SLM":
        plt.savefig("plot_mapping_functions/SLM-MF.png", dpi=300)
    if name == "MSLM":
        plt.savefig("plot_mapping_functions/MSLM-MF.png", dpi=300)
    if name == "Klobuchar":
        plt.savefig("plot_mapping_functions/Klobuchar-MF.png", dpi=300)
    if name == "SLMMSLM":
        plt.savefig("plot_mapping_functions/SLMMSLM-MF.png", dpi=300)
    if name == "all":
        plt.savefig("plot_mapping_functions/all-MF.png", dpi=300)

    

def main():
    name = "MSLM"    # SLM MSLM Klobuchar SLMMSLM all

    plot_mf(name)

if __name__ == '__main__':
    main()