"""
Author: Martijn Eppenga
Date: 20-02-2021
"""

import os
import sys
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import keyboard  

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
class spectogram(object):

    def __init__(self, fs, max_display_freq, stop = 'q', time_interval = 0.1, buffer_size =1, device=sd.default.device):
        """
        Arguments:
            fs: int or float, sample frequency in Hz
            max_display_freq: int or float, max frequency to plot
                in Hz
            stop: str, when this key is hit the program will stop
                default is q
            time_interval: float, time interval between updating
                plots in seconds (default is 0.1) 
            buffer_size: float, number of seconds of data that will
                be shown in a plot
            device:  Device index(es) or query string(s) specifying the device(s) to be used
                default sd.default.device
        """
        self.fs = fs
        if max_display_freq > fs/2:
            self.max_display_freq = fs
        else:
            self.max_display_freq = max_display_freq

        if time_interval >= buffer_size:
            raise Exception('Buffer size must be larger than time_interval')
        self.key_stop = stop

        # init buffer
        self.blocksize = int(fs * time_interval)
        self.buffer   = np.zeros(int(fs*buffer_size))
        self.old_buffer_data = np.zeros(self.blocksize)

        # create figure handels
        self.fig, (self.ax1, self.ax2) = plt.subplots(1,2)
        self.fig.canvas.mpl_connect('close_event', self.on_close_fig)
        self.ax1.set_ylim(-2, 2)
        self.ax2.set_ylim(0, 0.1)
        self.ax1.set_xlabel('Time [s]')
        self.ax2.set_xlabel('Frequency [Hz]')

        # create time and frequency x-axis
        self.time = np.arange(0, 1/fs*len(self.buffer), 1/fs)[0:len(self.buffer)]
        self.freq = np.fft.fftfreq(self.blocksize , 1/self.fs)[0:int(self.blocksize /2)]
        self.fft_index = np.nonzero(self.freq  <= self.max_display_freq)
        self.freq = self.freq[self.fft_index]
        self.fft_data = np.fft.fft(np.zeros(self.blocksize))[self.fft_index]
        self.window = self.hanning_VA(len(self.fft_data))

        # create plot handels
        self.ax1_plot_handle = self.ax1.plot(self.time, self.buffer)[0]
        self.ax2_plot_handle = self.ax2.plot(self.freq, np.abs(self.fft_data))[0]

        # set plots to full screen
        self.plt_set_fullscreen()
        plt.show(block=False)

        # create audio recoding object
        self.stream = sd.InputStream(channels=1, callback=self.plot_data,
                        blocksize=int(fs * time_interval), samplerate=self.fs,
                        device = device)
        
        self.is_running = False
        self.figure_callback_pause = 0.01 if time_interval > 0.01 else time_interval

        

            
    def update_bufffer(self, data):
        """
        Update the time domain audio buffer
        """
        N = len(self.buffer)
        M = len(data)
        if M > N:
            self.old_buffer_data = self.buffer
            self.buffer[:] = data[0:N]
        else:
            self.old_buffer_data = self.buffer[0:M].copy()
            self.buffer[0:N-M] = self.buffer[M:]
            self.buffer[N-M:]  = data

    def update_fft_data(self, new_data, old_data):
        """
        Update frequency domain audio data
        """
        self.fft_data[:] += self.fft(new_data-old_data, True)[self.fft_index]


    def fft(self, data, use_window=False):
        if use_window:
            if hasattr(self,'window'):
                if len(self.window) == len(data):
                    pass
                else:
                    self.window = self.hanning_VA(len(data))
            else:
                self.window = self.hanning_VA(len(data))
            data *= self.window
        return np.fft.fft(data)[0:int(len(data)/2)]/len(data)

    @staticmethod
    def hanning_VA(N):
        """Returns the hanning window defined by visual studio
        Calculates the N+1 point hanning window and retuns the first N points
        This implementation ensures a perfect periodic extension for spectral
        analysis using the fft
        Note the window thus has length N+1 and only the first N points are returned
        function: w(n) = 0.8164 * (1- cos(2pi * n / (N-1))
        Arguments:
            N: int, number of points of the hanning window
        """
        if not isinstance(N, (int, np.integer)):
            raise TypeError("N must be an integer")
        if N <= 0:
            raise ValueError("N must be a positive non zero integer")
        return 0.5 * (1 - np.cos(2*np.pi/N*np.arange(N)))

    def plot_data(self, indata, frames, time, status):
        try:
            if any(indata):
                self.update_bufffer(indata[:,0])
                self.update_fft_data(indata[:,0], self.old_buffer_data)
                self.update_plot()
            else:
                print('no input')
        except:
            print(sys.exc_info())
    
    def update_plot(self):
        """
        Update y data plots
        """
        self.ax1_plot_handle.set_ydata(self.buffer)
        self.ax2_plot_handle.set_ydata(np.abs(self.fft_data))
 

    def plt_set_fullscreen(self):
        """
        set figure size to full screen
        """
        # copy from Martin, set current image to full screen

        backend = str(plt.get_backend())
        mgr = plt.get_current_fig_manager()
        if backend == 'TkAgg':
            if os.name == 'nt':
                mgr.window.state('zoomed')
            else:
                mgr.resize(*mgr.window.maxsize())
        elif backend == 'wxAgg':
            mgr.frame.Maximize(True)
        elif backend == 'Qt4Agg':
            mgr.window.showMaximized()
        elif backend == 'GTK3Agg':
            mgr.window.maximize()

    def start(self):
        """
        Start recoding and displaying audio signal
        """
        print('Start recoding')
        self.is_running = True
        self.stream.start()
        # start thread to terminate recoding
        # when specific key is pressed
        self.thread = threading.Thread(target=self.thread_function, daemon=True)
        self.thread.start()
        
        while self.is_running:
            try:
                # update figure
                self.fig.canvas.draw()
                plt.pause(self.figure_callback_pause)
            except:
                print(sys.exc_info())
                time.sleep(0.01)

    def stop(self):
        """
        Stop recoding
        Terminate thread
        Close figure
        """
        print('Stop recoding')
        self.stream.stop()
        self.is_running = False
        plt.close(self.fig)
        # self.thread.join()

    def on_close_fig(self, event):
        """
        Stop recoding when figure is closed
        """
        if self.is_running:
            self.stop()

    def thread_function(self):
        """
        Terminate recoding after key press self.key_stop
        """
        while self.is_running:
            if keyboard.is_pressed(self.key_stop):
                # stop command detected
                self.stop()
                # terminate thread
            time.sleep(0.01)
    
if __name__ == '__main__':
    audio_analyzer = spectogram(8000, 1000, buffer_size=5, time_interval=0.1)
    audio_analyzer.start()