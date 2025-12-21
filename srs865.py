# --------------------------------------------------------------------
# Description: This module is used to read and analyze the data from LS372.
# --------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from . import inductor
from . import capacitor

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
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
    Data class to read the data from the files and store them in the class variables.
    Class Variables:
        input: np.array: The input voltage.
        frequncy: np.array: frequency measured.
        voltage: np.array: voltage measured.
        phase: np.array: phase measured.
        title: str: The title of the plot.
        inductance_expected: float: The expected inductance of the test component.
        capacitance_expected: float: The expected capacitance of the test component.
    '''
    def __init__(self,filepath,title,oscilation_freq=1e6,ignore_first=0,ignore_last=-1):
        '''
        Read the data from the file and store them in the class variables.
        input: 
            filepath: str: The path of the file.
            title: str: The title of the plot.
            oscilation_freq: float: Expect oscilation frequency of the test component.
        '''
        data = np.loadtxt(filepath, skiprows=1)[ignore_first:ignore_last].T
        self.input = data[0]
        self.frequency = data[1]
        self.voltage = data[2]
        self.phase = data[3] - data[3][data[2].argmin()]
        self.title = title
        self.inductance_expected = 3.3257e-6
        self.capacitance_expected = 1/(4*np.pi**2*oscilation_freq**2*self.inductance_expected)
        self.analysis = self.analysis(self)
    
    class analysis():

        def __init__(self,data):
            self.data = data

        def impedance(self,R_s=2064,filepath=None,mode='dif',plot=True):
            '''
            Calculate and plot the measured impedance curve.
            Model used: Z = (U/U_M-1)^-1 R_shunt
            '''
            if mode == 'dif':
                self.data.impedance = ((self.data.input*2/(self.data.voltage*np.exp(np.deg2rad(self.data.phase)*1j))-1)**-1)*R_s
            
            elif mode == 'abs':
                self.data.impedance = ((self.data.input/(self.data.voltage*np.exp(np.deg2rad(self.data.phase)*1j))-1)**-1)*R_s

            if plot:
                fig,axes = plt.subplots(2,1,sharex=True)

                axes[0].plot(self.data.frequency/1e3,np.abs(self.data.impedance))
                axes[1].plot(self.data.frequency/1e3,np.rad2deg(np.angle(self.data.impedance)))

                fig.tight_layout()

                axes[0].set_ylabel('Z [$\\rm \\Omega$]')
                axes[1].set_ylabel('Phase [$deg$]')
                axes[1].set_xlabel('Frequency [kHz]')

                axes[1].set_xscale('log')
                axes[0].set_yscale('log')

                axes[0].set_title(self.data.title)
                
                if filepath:
                    plt.savefig(filepath)
            
                return fig,axes
        
        def resistance(self,filepath=None):
            '''
            Calculate and plot the measured resistance curve.
            '''
            try:
                self.data.resistance = np.real(self.data.impedance)
            except:
                self.impedance()
                self.data.resistance = np.real(self.data.impedance)

            plt.figure()

            plt.plot(self.data.frequency/1e3,self.data.resistance,label='Measured Resistance') # Plot the measured resistance

            # plt.xscale('log')
            # plt.yscale('log')

            plt.xlabel('Frequency [kHz]')
            plt.ylabel('Resistance [Ohm]')

            plt.title(self.data.title)

            return plt.gcf()
        
        def reactance(self,filepath=None):
            '''
            Calculate and plot the measured reactance curve.
            '''
            try:
                self.data.reactance = np.imag(self.data.impedance)
            except:
                self.impedance()
                self.data.reactance = np.imag(self.data.impedance)

            plt.figure()

            plt.plot(self.data.frequency/1e3,self.data.reactance,label='Measured Reactance')

            # plt.xscale('log')
            # plt.yscale('log')

            plt.xlabel('Frequency [kHz]')
            plt.ylabel('Reactance [Ohm]')

            plt.title(self.data.title)

            return plt.gcf()
        
        def inductance(self,filepath=None):
            '''
            Calculate and plot the measured inductance curve.
            '''
            try:
                self.data.inductance = self.data.reactance/(2*np.pi*self.data.frequency)
            except:
                self.reactance()
                self.data.inductance = self.data.reactance/(2*np.pi*self.data.frequency)

            plt.figure()

            plt.hlines(self.data.inductance_expected,self.data.frequency[0]/1e3,self.data.frequency[-1]/1e3,linestyles='dashed',colors='r',label='Designed Inductance') # Plot the designed inductance
            
            plt.plot(self.data.frequency/1e3,self.data.inductance,label='Measured Inductance')

            # plt.xscale('log')
            # plt.yscale('log')

            plt.xlabel('Frequency [kHz]')
            plt.ylabel('Inductance [H]')

            plt.title(self.data.title)

            return plt.gcf()
        
        def capacitance(self,filepath=None):
            '''
            Calculate and plot the measured capacitance curve.
            '''
            try:
                self.data.capacitance = -1/(self.data.reactance*2*np.pi*self.data.frequency)
            except:
                self.reactance()
                self.data.capacitance = -1/(self.data.reactance*2*np.pi*self.data.frequency)

            plt.figure()

            plt.hlines(self.data.capacitance_expected,self.data.frequency[0]/1e3,self.data.frequency[-1]/1e3,linestyles='dashed',colors='r',label='Designed Capacitance')

            plt.plot(self.data.frequency/1e3,self.data.capacitance,label='Measured Capacitance')

            # plt.xscale('log')
            # plt.yscale('log')

            plt.xlabel('Frequency [kHz]')
            plt.ylabel('Capacitance [F]')

            plt.title(self.data.title)

            return plt.gcf()
        
        ## 整个复数域的拟合来!