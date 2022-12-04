
# Author: Satvik Dixit, Daniel M. Low

import numpy as np
import scipy.signal
import scipy.io.wavfile
import math
import soundfile as sf
import numpy.matlib

# function for computing cepstral peak prominence


def cpp(x,fs, normOpt='line', dBScaleOpt=True, statistic='median'):
	"""
	Computes cepstral peak prominence for a given x
	Parameters
	-----------
	x: ndarray
		The audio x
	fs: integer
		The sampling frequency
	normOpt: string
		'line', 'mean' or 'nonorm' for selecting normalisation type
	dBScaleOpt: binary
		True or False for using decibel scale
	Returns
	-----------
	cpp: ndarray
		The CPP with time values
	"""


	# Settings
	frame_length = int(np.round_(0.04*fs))
	frame_shift = int(np.round_(0.01*fs))
	half_len = int(np.round_(frame_length/2))
	x_len = len(x)
	frame_len = half_len*2 + 1
	NFFT = 2**(math.ceil(np.log(frame_len)/np.log(2)))
	quef = np.linspace(0, frame_len/1000, NFFT)

	# Allowed quefrency range
	pitch_range = [60, 333.3]
	quef_lim = [int(np.round_(fs/pitch_range[1])),
	            int(np.round_(fs/pitch_range[0]))]
	quef_seq = range(quef_lim[0]-1, quef_lim[1])

	# Time samples
	time_samples = np.array(
		range(frame_length+1, x_len-frame_length+1, frame_shift))
	N = len(time_samples)
	frame_start = time_samples-half_len
	frame_stop = time_samples+half_len

	# High-pass filtering
	HPfilt_b = [1 - 0.97]
	x = scipy.signal.lfilter(HPfilt_b, 1, x)

	# Frame matrix
	frameMat = np.zeros([NFFT, N])
	for n in range(0, N):
		frameMat[0: frame_len, n] = x[frame_start[n]-1:frame_stop[n]]

	# Hanning
	def hanning(N):
		x = np.array([i/(N+1) for i in range(1, int(np.ceil(N/2))+1)])
		w = 0.5-0.5*np.cos(2*np.pi*x)
		w_rev = w[::-1]
		return np.concatenate((w, w_rev[int((np.ceil(N % 2))):]))
	win = hanning(frame_len)
	winmat = np.matlib.repmat(win, N, 1).transpose() #np.tile(win, (N, 1)).transpose()
	frameMat = frameMat[0:frame_len, :]*winmat

	# Cepstrum
	SpecMat = np.abs(np.fft.fft(frameMat, axis=0))
	SpecdB = 20*np.log10(SpecMat)
	if dBScaleOpt:
		ceps = 20*np.log10(np.abs(np.fft.fft(SpecdB, axis=0)))
	else:
		ceps = 2*np.log(np.abs(np.fft.fft(SpecdB, axis=0)))

	# Finding the peak
	ceps_lim = ceps[quef_seq, :]
	ceps_max = ceps_lim.max(axis=0)
	max_index = ceps_lim.argmax(axis=0)

	# Normalisation
	ceps_norm = np.zeros([N])
	if normOpt == 'line':
		for n in range(0, N):
			p = np.polyfit(quef_seq, ceps_lim[:, n], 1)
			ceps_norm[n] = np.polyval(p, quef_seq[max_index[n]])
	elif normOpt == 'mean':
		ceps_norm = np.mean(ceps_lim)

	cpp = ceps_max-ceps_norm

	if statistic == 'median':
		cpp = np.median(cpp)

	return cpp, time_samples


def cpp_from_paths(paths, statistic = 'median'):
	'''

	Args:
		paths:
			list of path to files
		statistic: bool or str
			False, 'median'

	Returns:

	'''

	cpp_statistics = []
	for path in paths:
		x, fs = sf.read(path)
		cpp_across_time_windows, time_samples = cpp(x, fs, 'line',dBScaleOpt=True, statistic=statistic)
		# cpp_all_files.append(cpp_across_time_windows)
		cpp_statistics.append(cpp_across_time_windows)
	return cpp_statistics


'''
import soundfile as sf
import os
import matplotlib.pyplot as plt
import glob
paths =  glob.glob('./audio/data/audio_samples/*')


cpp_all_files = []

cpp_statistics = []


cpp_across_time_windows = cpp_from_paths(paths, statistic = "median")



plt.plot(cpp_across_time_windows)
labels = [n.split('/')[-1] for n in paths]
plt.xticks(labels=labels, ticks = range(len(labels)), rotation=45)
plt.tight_layout()
	
	
	
	




'''