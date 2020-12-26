import os
import glob
import librosa
import librosa.display
import matplotlib
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from matplotlib.colors import colorConverter
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, Button, HBox, VBox
import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np
import multiprocessing
from multiprocessing import Pool

from IPython.display import HTML
import cv2
import matplotlib.gridspec as gridspec
import tqdm

import warnings
warnings.filterwarnings("ignore")

#matplotlib.use('Agg')

# out = widgets.Output()

base_colour = colorConverter.to_rgba('white')
alphas = np.linspace(0, 1, 256 + 3)

temporal = colorConverter.to_rgba('red')
spectral = colorConverter.to_rgba('blue')

# make the colormaps
temporal_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',[base_colour,temporal],256)
temporal_cmap._init()  # create the _lut array, with rgba values
temporal_cmap._lut[:, -1] = alphas  # apply transparency

spectral_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',[base_colour,spectral],256)
spectral_cmap._init()  # create the _lut array, with rgba values
spectral_cmap._lut[:, -1] = alphas


# @out.capture()
def render_video(results_path, sig=2.0, asig=0.25, trim_first_frame=True):

	sig = list(np.arange(0, 4.25, 0.25)).index(sig)
	asig = list(np.arange(0, 4.25, 0.25)).index(asig)
	if trim_first_frame:
		offset = slice(1,-1)
	else:
		offset = slice(0,None)

	# Only load the first 5 seconds, otherwise size is unmanageable
	video_input = np.load(os.path.join(results_path,"video_input.npy"))
	audio_input = np.load(os.path.join(results_path, "audio_input.npy"))

	video_temporal_relevance = np.load(os.path.join(results_path, "video_temporal_relevance.npy")).astype('uint8')
	video_spatial_relevance = np.load(os.path.join(results_path, "video_spatial_relevance.npy")).astype('uint8')

	audio_temporal_relevance = np.load(os.path.join(results_path, "audio_temporal_relevance.npy"))
	audio_spectral_relevance = np.load(os.path.join(results_path, "audio_spectral_relevance.npy"))

	video_input = [frame for clip in video_input[:][:5] for frame in clip[offset]]
	video_temporal_relevance = video_temporal_relevance[:, sig, ...]
	video_temporal_relevance = [frame for clip in video_temporal_relevance[:5] for frame in clip[offset]]
	video_spatial_relevance = video_spatial_relevance[:, sig, ...]
	video_spatial_relevance = [frame for clip in video_spatial_relevance[:5] for frame in clip[offset]]

	prediction = glob.glob(f"{results_path}/pred_*")[0][len(results_path)+5:]

	# For some reason the scaling is really off, so multiply by 10e7 to return to normal.
	audio_input = librosa.amplitude_to_db(audio_input[...,0] * 10e7)
	spectrogram_rendering_config = {
		'y_axis': 'mel',
		'x_axis': 's',
		'sr': 16000,
		'fmin': 125,
		'fmax': 7500,
		'hop_length': 160
	}

	fig = plt.figure(constrained_layout=False, figsize=(12, 8))
	gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

	#video_data = zip(video_input, video_temporal_relevance, video_spatial_relevance)

	# Cut relevance maps downn to length
	video_temporal_relevance = video_temporal_relevance[0:len(video_input)]
	video_spatial_relevance = video_spatial_relevance[0:len(video_input)]

	video_temporal_relevance = np.asarray(video_temporal_relevance)
	video_spatial_relevance = np.asarray(video_spatial_relevance)

	video_temporal_combined = np.sum(video_temporal_relevance, axis=0)
	video_spatial_combined = np.sum(video_spatial_relevance, axis=0)

	print(audio_input.shape)
	#audio_input = audio_input[2]
	#audio_input = np.concatenate((audio_input[0], audio_input[1], audio_input[2], audio_input[3]), axis=1)
	audio_input = np.concatenate([audio for audio in audio_input[:][:5]], axis=1)
	#audio_input = np.reshape(audio_input, (64, -1))
	# audio_input = audio_input[0]
	#print(audio_input.shape)



	video_input = cv2.cvtColor(video_input[10], cv2.COLOR_BGR2GRAY)
	v_ax = fig.add_subplot(gs[0, 0])
	v_ax.imshow(video_input, cmap='gray')
	v_ax.axis('off')
	v_ax.set_title("Input video")


	v_ax2 = fig.add_subplot(gs[0, 1])
	v_ax2.imshow(video_input, cmap="gray")
	v_ax2.imshow(video_temporal_combined, alpha=0.5)
	v_ax2.axis('off')
	v_ax2.set_title("Temporal relevance")

	v_ax3 = fig.add_subplot(gs[0,2])
	v_ax3.imshow(video_input, cmap="gray")
	v_ax3.imshow(video_spatial_combined, alpha=0.5)
	v_ax3.axis('off')
	v_ax3.set_title("Spatial relevance")

	a_ax = fig.add_subplot(gs[1,0])
	librosa.display.specshow(audio_input, **spectrogram_rendering_config)
	#plt.colorbar()
	a_ax.set_title("Audio Input")

	# =*= Temporal relevance =*=
	a_ax2 = fig.add_subplot(gs[1,1])
	librosa.display.specshow(
		audio_input,
		cmap='gist_gray',
		**spectrogram_rendering_config)
	w = np.array([ 0.07, 0.72,  0.21])
	relevance_spectrogram = np.dot(audio_temporal_relevance[..., :3], w)
	relevance_spectrogram = relevance_spectrogram[:, asig, ...]
	#relevance_spectrogram = relevance_spectrogram[2]
	#relevance_spectrogram = np.concatenate((relevance_spectrogram[0], relevance_spectrogram[1], relevance_spectrogram[2], relevance_spectrogram[3]), axis=1)
	relevance_spectrogram = np.concatenate([audio for audio in relevance_spectrogram[:][:5]], axis=1)
	print(relevance_spectrogram.shape)
	#print(audio_temporal_relevance.shape)
	librosa.display.specshow(
		relevance_spectrogram,
		cmap=temporal_cmap,
		**spectrogram_rendering_config,
		vmax=1)
	a_ax2.set_title("Temporal relevance")

	# =*= Spectral relevance =*=
	a_ax3 = fig.add_subplot(gs[1,2])
	librosa.display.specshow(
		audio_input,
		cmap='gist_gray',
		**spectrogram_rendering_config)
	w = np.array([ 0.07, 0.72,  0.21])
	relevance_spectrogram = np.dot(audio_spectral_relevance[..., :3], w)
	relevance_spectrogram = relevance_spectrogram[:, asig, ...]
	#relevance_spectrogram = relevance_spectrogram[2]
	#relevance_spectrogram = np.concatenate((relevance_spectrogram[0], relevance_spectrogram[1], relevance_spectrogram[2], relevance_spectrogram[3]), axis=1)
	relevance_spectrogram = np.concatenate([audio for audio in relevance_spectrogram[:][:5]], axis=1)
	librosa.display.specshow(
		relevance_spectrogram,
		cmap=spectral_cmap,
		**spectrogram_rendering_config,
		vmax=1)
	a_ax3.set_title("Spectral relevance")

	fig.tight_layout()
	plt.show()


if __name__ == "__main__":
	#results = glob.glob("./results/*/*/")
	#results = ["./results/Hammering/_DSC0507/"]
	#results = ["./results/Bowling/v_Bowling_g12_c03/"]
	#results = ["./results/CliffDiving/v_CliffDiving_g16_c01/"]
	results = ["./results/CliffDiving/v_CliffDiving_g14_c05"]
	results = ["./results/CliffDiving/v_CliffDiving_g19_c04"]
	results = ["./results/CliffDiving/v_CliffDiving_g22_c05"]
	results = ["./results/Hammering/v_Hammering_g09_c03"]
	results = ["./results3/CliffDiving/v_CliffDiving_g10_c06"]
	results = ["./results4/CliffDiving/v_CliffDiving_g10_c06"]
	#results = ["./results3/BoxingPunchingBag/v_BoxingPunchingBag_g24_c02"]
	#results = ["./results4/BoxingPunchingBag/v_BoxingPunchingBag_g24_c02"]
	results = ["./results5/Archery/v_Archery_g08_c02"]
	results = ["./results5/Archery/v_Archery_g18_c01"]
	results = ["./results5/Archery/v_Archery_g18_c05"]
	results = ["./results5/CliffDiving/v_CliffDiving_g06_c02"]
	results = ["./results5/Hammering/v_Hammering_g15_c05"]
	#results = ["./results5/Hammering/v_Hammering_g18_c01"]
	results = ["./results5/Hammering/v_Hammering_g19_c02"]
	results = ["./results5/FrontCrawl/v_FrontCrawl_g10_c04"]
	results = ["./resultsFYP/CliffDiving/v_CliffDiving_g17_c05"]

	#with Pool(multiprocessing.cpu_count()) as p:
	#	tqdm.tqdm(p.imap(render_video, results), total=len(results))
	for i in tqdm.tqdm(range(len(results))):
		render_video(results[i])
		print(results[i])
	#render_video(results[0])
