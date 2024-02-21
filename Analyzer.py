# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:23:34 2024

@author: mauch
"""

import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from scipy.optimize import curve_fit
from scipy import constants
import math

DSP = 0
zero_energy = 17.4
# Accessing Planck constant
planck_constant = constants.Planck
Angles = 0, 25, 45, 45, 65, 85, 105, 125, 145


def GaussModel(x, a1, x10, sigma1):
    y_gauss = a1 * np.exp(-(x - x10) ** 2 / (2 * sigma1 ** 2))
    return y_gauss


plt.close('all')


def calibration():
    # Ask user to select folder
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
    file = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    file_path = os.path.join(folder_path, file[0])
    df = pd.read_excel(file_path)
    columns = df.columns
    a1_est = 1
    x10_est = 340
    sigma1_est = 5

    p0 = [a1_est, x10_est, sigma1_est]

    # x and y contain the full data columns
    x = df['Channel'][DSP:]
    y = df['Calibration'][DSP:] / max(df['Calibration'][DSP:])

    # do the fit
    popt, pcov = curve_fit(GaussModel, x, y, p0)

    a1_fit = popt[0]
    x10_fit = popt[1]
    sigma1_fit = popt[2]

    plt.figure()
    plt.scatter(x, y, label="Data")  # Corrected 'lable' to 'label'
    plt.plot(x, GaussModel(x, a1_fit, x10_fit, sigma1_fit),  # Corrected 'line' to 'plot'
             color="red",
             linestyle="dashed",  # Changed 'line_dash' to 'linestyle'
             label="Optimised curve")  # Changed 'legend_label' to 'label'
    plt.legend()  # Added legend to display labels
    plt.show()
    calibration_factor = zero_energy / x10_fit

    return calibration_factor, columns, folder_path, df


def plot_csv_files(folder_path, columns, df, calibration_factor, normalize=True):
    peak_indices = []
    peaks_ = []
    plt.figure(figsize=(10, 6))
    df['Channel'] = df['Channel'] * calibration_factor
    df['Channel_2'] = df['Channel'][DSP:]

    for i, col in enumerate(columns[1:]):
        x = df['Channel_2'][:len(df[col])]
        y = df[col][:len(df[col])] / max(df[col][:len(df[col])])

        if normalize:
            plt.plot(x, y, label=col)
        else:
            plt.plot(x, df[col][:len(df[col])], label=col)

        max_index = df[col].idxmax()
        peaks, _ = find_peaks(df[col])
        peak_indices.append(peaks)

        # Peak A
        a1_est = 1
        x10_est = zero_energy
        sigma1_est = 3

        # Define list of starting parameters
        p0 = [a1_est, x10_est, sigma1_est]

        # do the fit
        popt, pcov = curve_fit(GaussModel, x, y, p0)

        a1_fit = popt[0]
        x10_fit = popt[1]
        sigma1_fit = popt[2]
        peaks_.append(x10_fit)

        plt.figure()
        plt.scatter(x, y, color='black')
        plt.plot(x, GaussModel(x, a1_fit, x10_fit, sigma1_fit),
                 color="red",
                 linestyle="dashed")
        plt.axvline(x=x10_fit, color='black', linestyle='dotted')
        plt.xlabel("Energy/kV")
        plt.ylabel("Counts")
        plt.title("Fitting a Gaussian to multiple peaks")
        plt.legend(["Data", "Optimised curve", "Centre Peak A"])
        plt.grid(True)
        plt.show()

    plt.xlabel('Energy/kV')
    plt.ylabel('Counts')
    plt.title('Counts vs Channel')
    plt.legend()
    plt.grid(True)
    plt.show()

    return peaks_, df


def Eout(E_in, Angles,peaks_,folder_path):
    theoretical_data = []
    mec = 511  # keV, energy of electron
    for angle in Angles:
        e_out = E_in / (1 + (E_in / mec) * (1 - np.cos(np.radians(angle))))
        theoretical_data.append(e_out)
    plt.figure()
    plt.scatter(Angles, theoretical_data, color='red', label='Theoretical data')
    plt.scatter(Angles, peaks_, color='blue', label='Measurement data')
    plt.plot(Angles, theoretical_data, color='black', linestyle='--')
    plt.plot(Angles, peaks_, color='black', linestyle='--')
    plt.xlabel("Angle of incidence [degree]")
    plt.ylabel("Scattered photon energy [keV]")
    plt.title("Photon energy = f(AOI)")
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path+'/Theory.png') 
    plt.show()

    return theoretical_data


calibration_factor, columns, folder_path, df = calibration()
peaks_, df = plot_csv_files(folder_path, columns, df, calibration_factor)
theoretical_data = Eout(zero_energy, Angles,peaks_,folder_path)
