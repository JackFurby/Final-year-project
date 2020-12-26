"""
MobileNet in PyTorch.
https://github.com/okankop/Efficient-3DCNNs/blob/master/models/mobilenet.py
https://github.com/marvis/pytorch-mobilenet/blob/master/main.py
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nnAudio import Spectrogram
import torch.optim as optim
import torchexplain


class MobileNetV1(nn.Module):
	def __init__(self, dims, input_channels, embedding_size=1000, train=True):
		self.dims = dims
		super(MobileNetV1, self).__init__()
		assert 1 <= dims <= 3

		if train:
			self.lib = nn
		else:
			self.lib = torchexplain

		opt_conv = [None, nn.Conv1d, self.lib.Conv2d, self.lib.Conv3d]
		opt_bn = [None, nn.BatchNorm1d, self.lib.BatchNorm2d, self.lib.BatchNorm3d]
		opt_pool = [None, nn.AvgPool1d, self.lib.AvgPool2d, self.lib.AvgPool3d]

		Conv = opt_conv[dims]
		BatchNorm = opt_bn[dims]
		self.AvgPool = opt_pool[dims]

		def conv_bn(inp, oup, stride):
			return nn.Sequential(
				Conv(inp, oup, 3, stride, padding=1, bias=False),
				BatchNorm(oup),
				self.lib.ReLU(inplace=True))

		def conv_dw(inp, oup, stride):
			return nn.Sequential(
				Conv(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
				BatchNorm(inp),
				self.lib.ReLU(inplace=True),

				Conv(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
				BatchNorm(oup),
				self.lib.ReLU(inplace=True))

		self.features = nn.Sequential(
			conv_bn(input_channels, 32, 2),
			conv_dw(32, 64, 1),
			conv_dw(64, 128, 2),
			conv_dw(128, 128, 1),
			conv_dw(128, 256, 2),
			conv_dw(256, 256, 1),
			conv_dw(256, 512, 2),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 512, 1),
			conv_dw(512, 1024, 2),
			conv_dw(1024, 1024, 1),
		)
		self.fc = self.lib.Linear(1024, embedding_size)

	def forward(self, x):
		x = self.features(x)
		avg_pool = self.AvgPool(x.data.size()[-self.dims:])
		x = avg_pool(x).view(-1, 1024)
		x = self.fc(x)
		return x


class AudioMobileNet(MobileNetV1):
	def __init__(self, embedding_size=1000, spect_parameters=None, device="cpu", mel=True, train=True):
		super().__init__(embedding_size=embedding_size, dims=2, input_channels=1, train=train)
		self.mel = mel  # melSpectrogram can also be done in the dataloader
		if mel:
			if spect_parameters:
				self.spec_layer = Spectrogram.MelSpectrogram(**spect_parameters)
			else:
				config = dict(
					sr=16000,
					n_fft=400,
					n_mels=64,
					hop_length=160,
					window="hann",
					center=False,
					pad_mode="reflect",
					htk=True,
					fmin=125,
					fmax=7500,
					device=device
					# output_format='Magnitude'
				)
				self.spec_layer = Spectrogram.MelSpectrogram(**config)

	def forward(self, x):
		if self.mel:
			x = self.spec_layer(x)
			x = x.view(x.size(0), 1, x.size(1), x.size(2))
		x = super().forward(x)
		return x


class AudioClassifier(MobileNetV1):
	def __init__(self, embedding_size=1000, train=True):
		super().__init__(embedding_size=embedding_size, dims=2, input_channels=1, train=train)

		if train:
			self.lib = nn
		else:
			self.lib = torchexplain

		self.classifier = self.lib.Linear(embedding_size, 600)

	def forward(self, x):
		x = super().forward(x)
		x = self.classifier(self.lib.ReLU()(x))
		return x


class VideoMobileNet(MobileNetV1):
	def __init__(self, embedding_size=1000, train=True):
		super().__init__(embedding_size=embedding_size, dims=3, input_channels=3, train=train)

	def forward(self, x):
		x = super().forward(x)
		return x


class VideoClassifier(MobileNetV1):
	def __init__(self, embedding_size=1000, train=True):
		super().__init__(embedding_size=embedding_size, dims=3, input_channels=3, train=train)

		if train:
			self.lib = nn
		else:
			self.lib = torchexplain

		self.classifier = self.lib.Linear(embedding_size, 51)

	def forward(self, x):
		x = super().forward(x)
		x = self.classifier(self.lib.ReLU()(x))
		return x


class SyncNet(nn.Module):
	"""Net for pre training"""
	def __init__(self, videoNet, audioNet):
		super(SyncNet, self).__init__()
		self.videoNet = videoNet
		self.audioNet = audioNet

	def forward_video(self, x):
		return self.videoNet(x)

	def forward_audio(self, x):
		return self.audioNet(x)

	def forward(self, frames, audio):
		videoOut = self.forward_video(frames)
		audioOut = self.forward_audio(audio)
		return videoOut, audioOut


class FusionNet(nn.Module):
	"""A blank fusion network for classification"""
	def __init__(self, embedding=1000, train=True):
		super(FusionNet, self).__init__()

		if train:
			self.lib = nn
		else:
			self.lib = torchexplain

		videoModel = VideoMobileNet(embedding_size=embedding, train=train)
		audioModel = AudioMobileNet(embedding_size=embedding, mel=False, train=train)

		self.videoFeatureSize = videoModel.fc.out_features
		self.audioFeatureSize = audioModel.fc.out_features

		self.videoModel = videoModel.features
		self.audioModel = audioModel.features

		self.videoFc = self.lib.Linear(1024, self.videoFeatureSize)
		self.audioFc = self.lib.Linear(1024, self.audioFeatureSize)

		#self.fc1 = self.lib.Linear(self.videoFeatureSize + self.audioFeatureSize, 512)
		#self.fc2 = self.lib.Linear(512, 51)
		self.fc1 = self.lib.Linear(self.videoFeatureSize + self.audioFeatureSize, 51)

	def forward(self, frames, audio):
		xVideo = self.videoModel(frames)
		videoKernelSize = xVideo.data.size()[-3:]
		avgVideoPool = self.lib.AvgPool3d(videoKernelSize, stride=videoKernelSize)
		xVideo = avgVideoPool(xVideo).view(-1, 1024)
		xVideo = self.videoFc(xVideo)

		xAudio = self.audioModel(audio)
		audioKernelSize = xAudio.data.size()[-2:]
		avgAudioPool = self.lib.AvgPool2d(audioKernelSize, stride=audioKernelSize)
		xAudio = avgAudioPool(xAudio).view(-1, 1024)
		xAudio = self.audioFc(xAudio)

		x = torch.cat((xVideo, xAudio), dim=1)
		x = self.fc1(self.lib.ReLU()(x))
		#x = self.fc2(self.lib.ReLU()(x))
		return x


class FusionNetLoaded(nn.Module):
	"""Net for classification.
	This version will load pre trained weights given audio and video paths"""
	def __init__(self, videoPath, audioPath, videoEmbedding=1000, audioEmbedding=1000, freeze=True, train=True, mel=True):
		super(FusionNetLoaded, self).__init__()

		if train:
			self.lib = nn
		else:
			self.lib = torchexplain

		# Loading pre trained weights
		preTrainedVideo = torch.load(videoPath, map_location=torch.device('cpu'))
		preTrainedAudio = torch.load(audioPath, map_location=torch.device('cpu'))

		#videoModel = VideoMobileNet(embedding_size=videoEmbedding, train=train)
		videoModel = VideoClassifier(embedding_size=videoEmbedding, train=train)
		#audioModel = AudioMobileNet(embedding_size=audioEmbedding, mel=mel, train=train)
		audioModel = AudioClassifier(embedding_size=audioEmbedding, train=train)

		videoModel.load_state_dict(preTrainedVideo['state_dict'])
		audioModel.load_state_dict(preTrainedAudio['state_dict'])

		self.videoFeatureSize = videoModel.fc.out_features
		self.audioFeatureSize = audioModel.fc.out_features

		# Set output size of subnets
		# This will be the same as the largest input subnet
		if self.videoFeatureSize > self.audioFeatureSize:
			self.audioFeatureSize = self.videoFeatureSize
		else:
			self.videoFeatureSize = self.audioFeatureSize

		if freeze:
			# Freeze video model
			for param in videoModel.parameters():
				param.requires_grad_(False)

			# Freeze audio model
			for param in audioModel.parameters():
				param.requires_grad_(False)

		# Set subnets (missing original fc layers on each)
		self.videoModel = videoModel.features
		self.audioModel = audioModel.features

		# Create new fc layers for each sub net
		self.videoFc = self.lib.Linear(1024, self.videoFeatureSize)
		self.audioFc = self.lib.Linear(1024, self.audioFeatureSize)

		#self.fc1 = self.lib.Linear(self.videoFeatureSize + self.audioFeatureSize, 512)
		#self.fc2 = self.lib.Linear(512, 51)
		self.fc1 = self.lib.Linear(self.videoFeatureSize + self.audioFeatureSize, 51)

	def forward(self, frames, audio):
		xVideo = self.videoModel(frames)
		videoKernelSize = xVideo.data.size()[-3:]
		avgVideoPool = self.lib.AvgPool3d(videoKernelSize, stride=videoKernelSize)
		xVideo = avgVideoPool(xVideo).view(-1, 1024)
		xVideo = self.videoFc(xVideo)

		xAudio = self.audioModel(audio)
		audioKernelSize = xAudio.data.size()[-2:]
		avgAudioPool = self.lib.AvgPool2d(audioKernelSize, stride=audioKernelSize)
		xAudio = avgAudioPool(xAudio).view(-1, 1024)
		xAudio = self.audioFc(xAudio)

		x = torch.cat((xVideo, xAudio), dim=1)
		x = self.fc1(self.lib.ReLU()(x))
		#x = self.fc2(self.lib.ReLU()(x))
		return x


if __name__ == '__main__':
	audio_model = AudioMobileNet()
	# input_var = [N,C,D,H,W]
	input_var = range(0, 16000)  # dummy audio clip
	output = audio_model(input_var)
	print(output.shape)

	video_model = VideoMobileNet()
	# input_var = [N,C,D,H,W]
	input_var = Variable(torch.randn(1, 3, 16, 112, 112))
	print(video_model(input_var).shape)
