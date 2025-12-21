# --------------------------------------------------------------------
# Description: This module is used to read and analyze the data from Zurich Instruments.
# --------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tables as tb

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths

from . import inductor
from . import capacitor
from .circuit_model import *

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
config = {
    "font.family":'serif',
    "font.size":12,
    "mathtext.fontset":'stix',
    "font.serif": ['STIXGeneral']
}
plt.rcParams.update(config)
cmap = ['#6F6F6F','#547BB4','#C0321A','#DD7C4F','#6C61AF']

class data():
    '''
    Data class to read the Lock-in measurement data from the files and store them in the class variables.
    Class Variables:
        input: np.array: The input voltage.
        frequncy: np.array: frequency measured.
        voltage: np.array: voltage measured.
        phase: np.array: phase measured.
        title: str: The title of the plot.
        inductance_expected: float: The expected inductance of the test component.
        capacitance_expected: float: The expected capacitance of the test component.
    '''
    def __init__(self,filepath,title,input,CLR,Rshunt=300e-6,sample_number='000', data_clear=True):
        '''
        Read the data from the file and store them in the class variables.
        input: 
            filepath: str: The path of the file.
            title: str: The title of the plot.
            sample_number: str: Wanted sample number in the file. Default is '000'.
            oscilation_freq: float: Expect oscilation frequency of the test component.
        '''
        file = tb.open_file(filepath,mode='r')

        self.input = input
        self.CLR = CLR
        self.Rshunt = Rshunt
        self.current = self.input / (self.Rshunt + self.CLR + 200) # 200 Ohm is the internal resistance of the twisted pair cable.
        self.frequency = file.root[sample_number]['dev2191']['demods']['0']['sample']['frequency'].read()
        self.voltage = file.root[sample_number]['dev2191']['demods']['0']['sample']['r'].read()
        self.phase = file.root[sample_number]['dev2191']['demods']['0']['sample']['phase'].read()
        # Drop NaN values
        self.frequency = self.frequency[~np.isnan(self.voltage)]
        self.voltage = self.voltage[~np.isnan(self.voltage)]
        self.phase = self.phase[~np.isnan(self.phase)]
        self.title = title
        self.inductance_expected = 3.3257e-6
        if data_clear:
            index_drop = np.where(self.voltage < 5e-6)
            self.frequency = np.delete(self.frequency,index_drop)
            self.voltage = np.delete(self.voltage,index_drop)
            self.phase = np.delete(self.phase,index_drop)
        
        file.close()
        self.analysis = self.analysis(self)

    class analysis():
        def __init__(self,data):
            self.data = data
        def plot_voltage(self):
            '''
            Plot the voltage data.
            '''
            plt.figure(figsize=(6,4))
            plt.plot(self.data.frequency*1e-6,self.data.voltage*1e6,'o',markersize=3,color=cmap[0])
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Voltage (uV)')
            plt.title(self.data.title)
            plt.tight_layout()
            plt.show()
        def fit_normed_lorenzian(self,hfir=100):
            '''
            Fit the data to a normalized lorenzian function.
            input:
                initial_parameters: list: The initial parameters for the fit. [f0, Ql, A]
                hfir: int: The number of points to be used for the fit.
            output:
                popt: list: The optimized parameters. [f0, Ql, A]
                pcov: 2D array: The covariance of the optimized parameters.
            '''
            def normed_lorenzian(f, f0, Ql, A, k, b):
                return A/(1 + (2*Ql*(f-f0)/f0)**2) + k*f + b

            idx = np.argmax(self.data.voltage)
            idx_min = max(0, idx - hfir//2)
            idx_max = min(len(self.data.frequency), idx + hfir//2)

            freq_fit = self.data.frequency[idx_min:idx_max]
            volt_fit = self.data.voltage[idx_min:idx_max]

            try:
                peaks,_ = find_peaks(self.data.voltage,prominence=self.data.voltage.max()/2)
                widths,width_height,left_ips,right_ips = peak_widths(self.data.voltage, peaks, rel_height=0.707)
                geuss_Q = self.data.frequency[peaks[0]]/(self.data.frequency[int(right_ips[0])]-self.data.frequency[int(left_ips[0])])
                geuss_f0 = self.data.frequency[peaks[0]]
                geuss_A = self.data.voltage[peaks[0]]
            except:
                print('Warning: Peak finding failed, using max point as guess.')
                geuss_f0 = self.data.frequency[idx]
                geuss_Q = 1000
                geuss_A = self.data.voltage[idx]

            popt, pcov = curve_fit(normed_lorenzian, freq_fit, volt_fit, p0=[geuss_f0,geuss_Q,geuss_A,0,0],bounds=([[geuss_f0*0.9,10,0,-np.inf,-np.inf],[geuss_f0*1.1,1e5,np.inf,np.inf,np.inf]]),method='dogbox')

            plt.figure(figsize=(6,4))
            plt.plot(self.data.frequency*1e-6,self.data.voltage*1e6,'o',markersize=3,label='Data',color=cmap[0])
            plt.plot(freq_fit*1e-6,normed_lorenzian(freq_fit,*popt)*1e6,'-',label='Fit',color=cmap[2])
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Voltage (uV)')
            plt.title(self.data.title)
            plt.legend()
            plt.tight_layout()
            plt.show()

            print(f'Fitted parameters:\nf0 = {popt[0]:.2f} Hz\nQl = {popt[1]:.2f}\nA = {popt[2]:.2e} V\nk = {popt[3]:.2e} V/Hz\nb = {popt[4]:.2e} V')

            self.data.f0 = popt[0]
            self.data.f0err = np.sqrt(pcov[0,0])
            self.data.Ql = popt[1]
            self.data.Qlerr = np.sqrt(pcov[1,1])

            return popt, pcov