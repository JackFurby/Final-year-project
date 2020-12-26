import torch
import models.mobilenet_v1 as mobilenets
import torch.nn as nn


def loadModelVideo(videoPath):
	"""This function will load in kenetics pre trainied model and convert its
	state_dict to our video net model.

	args:
		videoPath: path to pre trainied video model (Pytorch model) as a string

	"""

	videoNet = mobilenets.VideoMobileNet(embedding_size=1000)
	refModel = {'state_dict': videoNet.state_dict()}  # contains correct param names

	videoModel = torch.load(videoPath, map_location=torch.device('cpu'))

	tempVideo = list(videoModel['state_dict'].keys())
	tempRef = list(refModel['state_dict'].keys())

	# Remain keys to match our model
	for i in range(len(tempVideo)):
		# One for audio, one for video
		# The one for audio probably needs replacing as its taking the same weights from the visual model
		refModel['state_dict'][tempRef[i]] = videoModel['state_dict'].get(tempVideo[i])  # Video

	videoNet.load_state_dict(refModel['state_dict'])

	return videoNet


def loadModelAudio(audioPath):
	"""This function will load in kenetics pre trainied model and convert its
	state_dict to our video net model.

	args:
		audioPath: path to pre trainied video model (Pytorch model) as a string

	"""

	audioNet = mobilenets.AudioMobileNet(embedding_size=1000)
	refModel = {'model_state_dict': audioNet.state_dict()}  # contains correct param names

	audioModel = torch.load(audioPath, map_location=torch.device('cpu'))

	tempAudio = list(audioModel['model_state_dict'].keys())
	tempRef = list(refModel['model_state_dict'].keys())

	# Remain keys to match our model
	for i in range(len(tempAudio)):
		splitRef = tempRef[i].split(".")
		if len(splitRef) > 2 and (splitRef[2] != 1 or splitRef[2] != 4):
			# One for audio, one for video
			# The one for audio probably needs replacing as its taking the same weights from the audible model
			refModel['model_state_dict'][tempRef[i]] = audioModel['model_state_dict'].get(tempAudio[i])  # Audio

	audioNet.load_state_dict(refModel['model_state_dict'])

	return audioNet


if __name__ == "__main__":

	videoNet = loadModelVideo("./saves/videoNetBest.pth")
	audioNet = loadModelAudio("./saves/audioNetBest.pth")
	#videoNet = loadModel("./saves/kinetics_mobilenet/kinetics_mobilenet_1.0x_RGB_16_best.pth")

	#torch.save({
	#    'model_state_dict': videoNet.state_dict()
	#}, './saves/kinetics_mobilenet_save.pth')

	torch.save({
		'model_state_dict': videoNet.state_dict()
	}, './saves/videoNetBestNew.pth')

	torch.save({
		'model_state_dict': audioNet.state_dict()
	}, './saves/audioNetBestNew.pth')
