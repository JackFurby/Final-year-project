import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from spatial_transforms import *
from temporal_transforms import *
from nnAudio import Spectrogram


def getMeanStdNorm():
	norm_value = 255
	mean = [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]
	std = [38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value]

	return mean, std, norm_value

def getTransforms():
	# -*- Video params -*-
	# www.github.com/okankop/Efficient-3DCNNs/blob/355eccbd0495237b562d8336e0ecf2e258db5c04/mean.py#L1

	mean, std, norm_value = getMeanStdNorm()

	initial_scale = 1.0
	n_scales = 5
	scale_step = 0.84089641525
	sample_size = 112
	sample_length = 16
	scales = [initial_scale]
	for i in range(1, n_scales):
		scales.append(scales[-1] * scale_step)

	## -*- transforms -*-
	audio_transform = ToTensorSpect()

	# train transforms
	train_temporal_transform = TemporalRandomCrop(sample_length, 1)
	train_spatial_transform = Compose([
		RandomHorizontalFlip(),
		MultiScaleRandomCrop(scales, sample_size),
		ToTensor(norm_value),
		Normalize(mean, std)
	])

	train_transforms = {
		'spatial_transform': train_spatial_transform,
		'temporal_transform': train_temporal_transform,
		'audio_transform': audio_transform
	}

	# val transforms
	val_temporal_transform = TemporalCenterCrop(sample_length, 1)
	val_spatial_transform = Compose([
		Scale(sample_size),
		CenterCrop(sample_size),
		ToTensor(norm_value),
		Normalize(mean, std)
	])

	val_transforms = {
		'spatial_transform': val_spatial_transform,
		'temporal_transform': val_temporal_transform,
		'audio_transform': audio_transform
	}

	# test transforms
	test_spatial_transform = Compose([
		Scale(int(sample_size / 1.0)),
		CornerCrop(sample_size, 'c'),
		ToTensor(norm_value),
		Normalize(mean, std)
	])

	test_transforms = {
		'spatial_transform': test_spatial_transform,
		'audio_transform': audio_transform
	}

	return train_transforms, val_transforms, test_transforms


class ToTensorSpect(object):
	def __init__(self):
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
			device="cpu")
		self.to_spec = Spectrogram.MelSpectrogram(**config)

	def __call__(self, audio_clip):
		x = torch.from_numpy(audio_clip).float()
		return self.to_spec(x)
