import os
from os import path
import glob
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas
import wandb


def main(root):
	classPaths = glob.glob(path.join(root + '/*'), recursive=True)
	classes = []
	meanAudio = []
	meanVideo = []
	medianAudio = []
	medianVideo = []
	stdAudio = []
	stdVideo = []
	maxAudio = []
	minAudio = []
	maxVideo = []
	minVideo = []
	audioValues = []
	videoValues = []
	for i in classPaths:
		relevanceFiles = glob.glob(path.join(i + '/**/*.ratio'), recursive=True)
		audio = []
		video = []
		for j in relevanceFiles:
			joinedString = j.split("/")[-1][:-6]
			split = joinedString.split("-")
			audioR = float(split[0][-5:])
			videoR = float(split[1][-5:])
			audio.append(audioR)
			video.append(videoR)

		classes.append(i.split("/")[-1])
		audioValues.append(audio)
		videoValues.append(video)

		meanAudio.append(round(statistics.mean(audio), 3))
		meanVideo.append(round(statistics.mean(video), 3))
		medianAudio.append(round(statistics.median(audio), 3))
		medianVideo.append(round(statistics.median(video), 3))
		stdAudio.append(statistics.stdev(audio))
		stdVideo.append(statistics.stdev(video))
		maxAudio.append(max(audio))
		minAudio.append(min(audio))
		maxVideo.append(max(video))
		minVideo.append(min(video))

	df = pandas.DataFrame({
		"classes": classes,
		"meanAudio": meanAudio,
		"meanVideo": meanVideo,
		"medianAudio": medianAudio,
		"medianVideo": medianVideo,
		"stdAudio": stdAudio,
		"stdVideo": stdVideo,
		"minAudio": minAudio,
		"maxAudio": maxAudio,
		"minVideo": minVideo,
		"maxVideo": maxVideo})
	#df.to_pickle("./processed_df.pkl")
	#wandb.save("./processed_df.pkl")

	audioDf = pandas.DataFrame(audioValues, index=classes)
	#audioDf.to_pickle("./audio_df.pkl")
	#wandb.save("./audio_df.pkl")

	videoDf = pandas.DataFrame(videoValues, index=classes)
	#videoDf.to_pickle("./video_df.pkl")
	#wandb.save("./video_df.pkl")
	"""
	df = pandas.read_pickle("./processed_df.pkl")
	audioDf = pandas.read_pickle("./audio_df.pkl")
	videoDf = pandas.read_pickle("./video_df.pkl")
	"""

	"""
	# List of accuracies per class (ordered by meanAudio relevance)
	below8 = [77, 52, 81, 25, 87, 62, 12, 100, 90, 41, 40, 42, 100, 68, 52, 57, 38, 43, 84, 89, 27, 43, 33]
	below95 = [93, 11, 35, 38, 42, 62, 56, 25, 47, 25, 18, 36, 12, 70, 43, 100]
	above95 = [31, 100, 30, 68, 11, 38, 20, 50, 22, 33, 31, 23]
	accuracyList = below8 + below95 + above95

	print(sum(below8) / len(below8))
	print(sum(below95) / len(below95))
	print(sum(above95) / len(above95))
	"""

	df = df.sort_values(by=["meanAudio"])
	# df['accuracy'] = accuracyList  # Add class accuracy to df
	audioDf = audioDf.T.reindex(audioDf.T.mean().sort_values().index, axis=1)
	videoDf = videoDf.T.reindex(videoDf.T.mean().sort_values(ascending=False).index, axis=1)
	print(df)

	width = 1.0

	# Audio / video rellevance
	fig, ax = plt.subplots()
	ax.barh(df.classes, df.meanAudio, width, color='DarkOrange', label='Mean Audio Relevance', edgecolor="black")
	ax.barh(df.classes, df.meanVideo, width, color='deepskyblue', left=df.meanAudio, label='Mean Video Relevance', edgecolor="black")

	ax.grid()
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
	ax.set_xlim([0, 1])
	ax.margins(x=0, y=0)

	plt.xticks(np.arange(0, 1, 0.1), ('0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'))
	plt.subplots_adjust(left=0.24, bottom=0.12, right=0.99, top=0.99)
	plt.show()

	"""
	# per class accuacy
	fig, ax = plt.subplots()
	ax.barh(df.classes, df.accuracy, width, color='mediumpurple', label='Per class accuracy', edgecolor="black")

	ax.grid()
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
	ax.set_xlim([0, 100])
	ax.margins(x=0, y=0)

	plt.xticks(np.arange(0, 100, 10), ('0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'))
	plt.subplots_adjust(left=0.24, bottom=0.12, right=0.99, top=0.99)
	plt.show()
	"""

	# Boxplots
	color = {
		'boxes': 'DarkOrange',
		'whiskers': 'DarkOrange',
		'medians': 'Black',
		'caps': 'Black'}


	boxplot = audioDf.plot.box(vert=False, grid=True, color=color)

	plt.subplots_adjust(left=0.23, bottom=0.03, right=0.99, top=0.99)
	plt.show()


	boxplot = videoDf.plot.box(vert=False, grid=True)

	plt.subplots_adjust(left=0.23, bottom=0.03, right=0.99, top=0.99)
	plt.show()


if __name__ == "__main__":
	#wandb.init(project="fusion-reproduction")
	#config = wandb.config
	main(root="./results")
