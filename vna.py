# --------------------------------------------------------------------
# Description: This module is used to read and analyze the data from VNA.
# --------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

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
    Data class to read VNA measurement data and store them in class variables.
    Class Variables:
        raw_data: pd.DataFrame: Raw data read from the file
        frequency: np.array: Frequency measured
        s21_scl: np.array: S21 magnitude
        s21_ang: np.array: S21 phase
        title: str: The title for plots
    '''
    def __init__(self, filepath, title, device='default'):
        '''
        Read the S-parameters from VNA data file
        Args:
            filepath: str: Path to the VNA data file
            title: str: Title for plots
            port_number: int: Number of ports (1 or 2)
        '''
        if device == 'default':
            raw_data = pd.read_csv(filepath,sep=';',header=2,encoding='GBK')
            raw_data.drop('Unnamed: 9',axis=1,inplace=True)
            raw_data.rename(columns={'db:Trc2_Z<-S21serOhm]': 'db:Trc2_Z<-S21[serOhm]', 'ang:Trc2_Z<-S21ser[癩': 'ang:Trc2_Z<-S21[serdeg]'},inplace=True)   
            self.frequency = raw_data['freq[Hz]']
            self.s21_scl = raw_data['db:Trc3_S21']
            self.s21_ang = raw_data['ang:Trc3_S21']
        elif device == 'new':
            raw_data = pd.read_csv(filepath,sep=',',header=5,skipfooter=2,encoding='GBK')
            self.frequency = raw_data['Freq(Hz)']
            try:
                self.s21_scl = raw_data['S21 Log Mag(dB)']
            except:
                self.s21_scl = raw_data['S12 Log Mag(dB)']

        self.title = title
        self.analysis = self.analysis(self)

    class analysis():
        def __init__(self, data):
            self.data = data

        def Q(self,lower_initial_offset=1e4,upper_initial_offset=1e4,peak_num=1):
            '''
            Calculate the Q factor of the resonator
            '''
            if peak_num == 1:
                f = interp1d(self.data.frequency,self.data.s21_scl -self.data.s21_scl.min() - 3)
                self.data.raw_Q = (self.data.frequency[self.data.s21_scl.argmin()]/np.abs((fsolve(f,self.data.frequency[self.data.s21_scl.argmin()] + upper_initial_offset) - fsolve(f,self.data.frequency[self.data.s21_scl.argmin()] - lower_initial_offset))))
                print(f'raw_Q = {self.data.raw_Q[0]:.2f}')
                return self.data.raw_Q
            else:
                peaks,__ = find_peaks(-self.data.s21_scl, prominence=10)

                f = []
                for i in range(peak_num):
                    f.append(interp1d(self.data.frequency,self.data.s21_scl -self.data.s21_scl[peaks[i]] - 3))

                self.data.raw_Q = []
                
                for i in range(peak_num):
                    self.data.raw_Q.append((self.data.frequency[peaks[i]]/np.abs((fsolve(f[i],self.data.frequency[peaks[i]] + upper_initial_offset[i]) - fsolve(f[i],self.data.frequency[peaks[i]] - lower_initial_offset[i])))))

                for i in range(peak_num):
                    print(f'raw_Q_{i+1} = {self.data.raw_Q[i][0]:.2f}')
                
                return self.data.raw_Q

        def plot_s21(self, filepath=None):
            '''
            Plot S21 magnitude (dB) and phase (degree)
            '''
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), dpi=100)
            
            # Plot magnitude
            axes[0].plot(self.data.frequency/1e6, self.data.s21_scl,label='S21')
            
            if 's21_ang' in self.data.__dict__:
                # Plot phase
                axes[1].plot(self.data.frequency/1e6, self.data.s21_ang, label='S21')

            axes[0].set_ylabel('Magnitude [dB]')
            axes[1].set_ylabel('Phase [deg]')
            axes[1].set_xlabel('Frequency [MHz]')
            # axes[0].legend()
            # axes[1].set_xscale('log')

            axes[0].set_title(self.data.title)
            
            fig.tight_layout(h_pad=0)

            if filepath:
                plt.savefig(filepath)
            
            return fig, axes
        
        def fit_s21_res_single(self, model='simple_lorentzian', initial_parameters=[6e6, 200, 3e-6], hfir=1000, filepath=None):
            
            if model == 'simple_lorentzian':
                argmin = self.data.s21_scl.argmin()

                popt, pcov = curve_fit(simple_lorentzian, self.data.frequency[argmin - hfir:argmin+hfir], self.data.s21_scl[argmin - hfir:argmin+hfir], p0=initial_parameters, bounds=[(0, 0, 0), (np.inf, np.inf, np.inf)])
                f0, Q, L = popt
                df0, dQ, dL = np.sqrt(np.diag(pcov))
                print(f'f0 = {f0:.2f} ± {df0:.2f} Hz')
                print(f'Q = {Q:.2f} ± {dQ:.2f}')
                print(f'L = {L:.3e} ± {dL:.3e} H')
                self.data.L = L
                self.data.dL = dL
                self.data.Q = Q
                self.data.dQ = dQ
                self.data.C = 1/(4*np.pi**2*f0**2*L)
                self.data.dC = np.sqrt((1/(4*np.pi**2*f0**2*L**2)*dL)**2 + (1/(2*np.pi*f0**3*L)*df0)**2)
                print(f'C = {self.data.C*1e9:.6f} ± {self.data.dC*1e9:.6f} nF')
                fit = simple_lorentzian(self.data.frequency, f0, Q, L)
            
            elif model == 'S21_res':
                def loss(params, x_data, mag_data, phase_data):
                    z = S21_res(x_data, *params)
                    model_mag = 20*np.log10(np.abs(z))
                    model_phase = np.angle(z,True)

                    # 处理相位周期性误差
                    phase_diff = model_phase - phase_data
                    phase_error = (phase_diff + 180) % (2 * 180) - 180

                    # 计算模长误差（in dB）
                    mag_error = model_mag - mag_data

                    # 总损失（可调整权重）
                    total_loss = np.sum(mag_error**2) + np.sum(phase_error**2)
                    return total_loss
                
                result = minimize(loss, initial_parameters, args=(self.data.frequency, self.data.s21_scl, self.data.s21_ang), method='Nelder-Mead', options={'maxiter': 20000}, bounds=((0, None), (0, None), (0, None), (0, None), (0, None), (0, None)))
                print(result.success)
                Z0, C, Rs, Rp, Ra, Rb = result.x
                print(f'Z0 = {Z0:.2f} Ohm')
                print(f'C = {C*1e9:.3f} nF')
                print(f'Rs = {Rs:.2f} Ohm')
                print(f'Rp = {Rp:.2f} Ohm')
                print(f'Ra = {Ra:.2f} Ohm')
                print(f'Rb = {Rb:.2f} Ohm')
                self.data.C = C
                self.data.Rs = Rs
                self.data.Rp = Rp
                fit = S21_res(self.data.frequency, Z0, C, Rs, Rp, Ra, Rb)
                fit_ang = np.rad2deg(np.angle(fit))
                fit = 20*np.log10(np.abs(fit))
            
            else:
                raise ValueError('Invalid model. Please choose from "simple_lorentzian" or "S21_res".')

            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), dpi=100)

            # Plot magnitude
            axes[0].plot(self.data.frequency/1e6, self.data.s21_scl, label='Data')
            if model == 'simple_lorentzian':
                axes[0].plot(self.data.frequency[argmin - hfir:argmin+hfir]/1e6, fit[argmin - hfir:argmin+hfir], label='Fit')
            else:
                axes[0].plot(self.data.frequency/1e6, fit, label='Fit')
            axes[0].set_ylabel('Magnitude [dB]')
            axes[0].legend()

            if 's21_ang' in self.data.__dict__:
                # Plot phase
                axes[1].plot(self.data.frequency/1e6, self.data.s21_ang, label='Data')
                if model == 'simple_lorentzian':
                    pass
                else:
                    axes[1].plot(self.data.frequency/1e6, fit_ang, label='Fit')
                axes[1].legend()
            axes[1].set_ylabel('Phase [deg]')
            axes[1].set_xlabel('Frequency [MHz]')
            # axes[1].set_xscale('log')

            axes[0].set_title(self.data.title)

            fig.tight_layout(h_pad=0)

            if filepath:
                plt.savefig(filepath)
            
            return fig, axes

        def fit_s21_res_multi(self, model='simple_lorentzian', initial_parameters=[[6e6, 200, 3e-6], [5e6,200,3e-6]], hfir=[1000,1000], filepath=None):
            
            if model == 'simple_lorentzian':
                # Find the frequency closest to 5e6
                peaks,__ = find_peaks(-self.data.s21_scl, prominence=10)

                argmins = []
                self.data.f0s = []
                self.data.Qs = []
                self.data.Ls = []
                self.data.Cs = []
                self.data.df0s = []
                self.data.dQs = []
                self.data.dLs = []
                self.data.dCs = []
                fits = []
                for i in range(len(initial_parameters)):
                    argmins.append(peaks[i])

                    popt, pcov = curve_fit(simple_lorentzian, self.data.frequency[argmins[i] - hfir[i]:argmins[i] + hfir[i]], self.data.s21_scl[argmins[i] - hfir[i]:argmins[i] + hfir[i]], p0=initial_parameters[i], bounds=[(0, 0, 0), (np.inf, np.inf, np.inf)])
                    
                    self.data.f0s.append(popt[0])
                    self.data.Qs.append(popt[1])
                    self.data.Ls.append(popt[2])
                    self.data.df0s.append(np.sqrt(np.diag(pcov))[0])
                    self.data.dQs.append(np.sqrt(np.diag(pcov))[1])
                    self.data.dLs.append(np.sqrt(np.diag(pcov))[2])
                    print(f'f0{i+1:d} = {self.data.f0s[i]:.2f} ± {self.data.df0s[i]:.2f} Hz')
                    print(f'Q{i+1:d} = {self.data.Qs[i]:.2f} ± {self.data.dQs[i]:.2f}')
                    print(f'L{i+1:d} = {self.data.Ls[i]:.3e} ± {self.data.dLs[i]:.3e} H')
                    
                    self.data.Cs.append(1/(4*np.pi**2*self.data.f0s[i]**2*self.data.Ls[i]))
                    self.data.dCs.append(np.sqrt((1/(4*np.pi**2*self.data.f0s[i]**2*self.data.Ls[i]**2)*self.data.dLs[i])**2 + (1/(2*np.pi*self.data.f0s[i]**3*self.data.Ls[i])*self.data.df0s[i])**2))
                    print(f'C{i+1:d} = {self.data.Cs[i]*1e9:.6f} ± {self.data.dCs[i]*1e9:.6f} nF')
                
                    fits.append(simple_lorentzian(self.data.frequency, self.data.f0s[i], self.data.Qs[i], self.data.Ls[i]))

            else:
                raise ValueError('Invalid model. Please choose "simple_lorentzian".')
            
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), dpi=100)

            # Plot magnitude
            axes[0].plot(self.data.frequency/1e6, self.data.s21_scl, label='Data')
            for i in range(len(fits)):
                axes[0].plot(self.data.frequency[argmins[i] - hfir[i]:argmins[i] + hfir[i]]/1e6, fits[i][argmins[i] - hfir[i]:argmins[i] + hfir[i]], label=f'Fit {i+1}')
            axes[0].set_ylabel('Magnitude [dB]')
            axes[0].legend()

            if 's21_ang' in self.data.__dict__:
                # Plot phase
                axes[1].plot(self.data.frequency/1e6, self.data.s21_ang, label='Data')
            axes[1].set_ylabel('Phase [deg]')
            axes[1].set_xlabel('Frequency [MHz]')
            # axes[1].legend()

            axes[0].set_title(self.data.title)

            fig.tight_layout(h_pad=0)

            if filepath:
                plt.savefig(filepath)
            
            return fig, axes