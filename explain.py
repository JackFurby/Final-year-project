import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models.mobilenet_v1 as mobilenets
import dataset
import wandb
import time
import os
import random
from spatial_transforms import *
from temporal_transforms import *
from transforms import *
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import selective_relevance as sr
from pathlib import Path
import argparse


base_colour = matplotlib.colors.colorConverter.to_rgba('black')
alphas = np.linspace(0, 1, 256 + 3)

temporal = matplotlib.colors.colorConverter.to_rgba('red')
spatial = matplotlib.colors.colorConverter.to_rgba('blue')

# make the colormaps
temporal_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', [base_colour, temporal], 256)
temporal_cmap._init()  # create the _lut array, with rgba values
temporal_cmap._lut[:, -1] = alphas  # apply transparency

spatial_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', [base_colour, spatial], 256)
spatial_cmap._init()  # create the _lut array, with rgba values
spatial_cmap._lut[:, -1] = alphas


# TODO: class level performance
def _per_video_performance(args, model, device, output, target):
	target = torch.tensor(target).to(device)
	pred = output.max(1, keepdim=True)[1]
	per_video_pred = torch.mode(pred.squeeze().float())[0]
	return per_video_pred.eq(target.view_as(per_video_pred))


# TODO: change 'video' to 'visual'
def _per_video_proportion_relevance(
	num_clips,
	inputs,
	gradients,
	num_video_features=1000,
	num_audio_features=1000):

	# Calculate the proportions
	vid_ins = torch.zeros([num_clips, num_video_features])
	vid_grads = torch.zeros([num_clips, num_video_features])

	aud_ins = torch.zeros([num_clips, num_audio_features])
	aud_grads = torch.zeros([num_clips, num_audio_features])

	for in_, grad_ in zip(inputs, gradients):
		in_ = in_.cpu()  # TO DO: fix this to work on GPU
		grad_ = grad_.cpu()  # TO DO: fix this to work on GPU
		vid_ins += in_[:, :num_video_features]
		vid_grads += grad_[:, :num_video_features]

		aud_ins += in_[:, num_video_features:]
		aud_grads += grad_[:, num_video_features:]

	vid_ins = vid_ins.sum(dim=0)
	vid_grads = vid_grads.sum(dim=0)

	aud_ins = aud_ins.sum(dim=0)
	aud_grads = aud_grads.sum(dim=0)

	aud_grad_ratio = aud_grads.sum() / aud_ins.sum()
	vid_grad_ratio = vid_grads.sum() / vid_ins.sum()

	proportion_audio_relevance = float(aud_grad_ratio / (aud_grad_ratio + vid_grad_ratio))
	proportion_video_relevance = float(vid_grad_ratio / (aud_grad_ratio + vid_grad_ratio))
	assert proportion_audio_relevance + proportion_video_relevance < 100

	return proportion_audio_relevance, proportion_video_relevance


def _clip_level_selective_relevance(video_clip, video_relevance, audio_clip, audio_relevance, mean=[0, 0, 0], std=[1, 1, 1]):

	for channel in range(video_relevance.shape[0]):
		# video_relevance[channel, ...] is (w, h)
		video_relevance[channel, ...] /= abs(video_relevance[channel, ...]).max()
	video_clip = video_clip.transpose(1, 2, 3, 0)  # (frames, w, h, c)

	# normalise
	for frame_idx in range(len(video_clip)):
		for channel in range(3):
			video_clip[frame_idx][..., channel] += mean[channel]
			video_clip[frame_idx][..., channel] *= std[channel]

	video_clip = [(video_clip[f] * 255).astype(np.uint8) for f in range(len(video_clip))]

	audio_clip = np.transpose(audio_clip, (1, 2, 0))[..., 0]
	audio_clip /= abs(audio_clip.max())
	audio_clip = (audio_clip * 255).astype(np.uint8)
	audio_relevance /= abs(audio_relevance).max()

	#sig_range = np.linspace(0, 4, 7) # FIXME: this can be more descriptive!!
	sig_range = list(np.arange(0, 4.25, 0.25))
	n_sigs = len(sig_range)

	temporal_visual_relevance = np.empty((n_sigs, 16, 112, 112, 3))
	spatial_visual_relevance = np.empty((n_sigs, 16, 112, 112, 3))
	temporal_audible_relevance = np.empty((n_sigs, 64, 96, 3))  # FIXME: should be 1 channel...
	spectral_audible_relevance = np.empty((n_sigs, 64, 96, 3))  # FIXME: should be 1 channel...

	for level, sig in enumerate(sig_range):
		audio_sr = sr.selective_relevance(
			audio_relevance.cpu(),
			sig=sig,
			img=True)
		video_sr = sr.selective_relevance(
			video_relevance.cpu(),
			sig=sig,
			img=False)

		temporal_visual_relevance[level] = np.array(video_sr[0])
		spatial_visual_relevance[level] = np.array(video_sr[1])
		temporal_audible_relevance[level] = np.array(audio_sr[0])
		spectral_audible_relevance[level] = np.array(audio_sr[1])

	return (
		temporal_visual_relevance,
		spatial_visual_relevance,
		temporal_audible_relevance,
		spectral_audible_relevance)


def _construct_test_batch(audio, video, sample_length, audio_sample_length, video_samples_per_second=16, audio_samples_per_second=96):
	num_frames = video.shape[1]
	num_clips = num_frames // video_samples_per_second
	audio_offset = audio_samples_per_second // video_samples_per_second

	# -*- construct video batch -*-
	video_input = []
	audio_input = []

	start, end = 0, sample_length
	for clip in range(num_clips):
		audio_clip = audio[:, :, slice(start * audio_offset, end * audio_offset)]
		video_clip = video[:, slice(start, end)]

		# The end of the clip -> throw it away
		if audio_clip.shape[2] < audio_sample_length:
			continue

		audio_input.append(audio_clip)
		video_input.append(video_clip)
		start = end
		end += sample_length

	video_input = torch.stack(video_input).requires_grad_()
	audio_input = torch.stack(audio_input).requires_grad_()

	return audio_input, video_input


# TODO: function that simulates live input, so generate an input block per frame of the video
def _construct_simulation_batch(audio, video):
	num_frames = video.shape[1]
	num_clips = num_frames // sample_length
	audio_offset = audio_sample_length // sample_length

	# -*- construct video batch -*-
	audio_input = torch.empty((num_clips, 1, 64, 96)).to(device).requires_grad_()
	video_input = torch.empty((num_clips, 3, 16, 112, 112)).to(device).requires_grad_()

	start, end = 0, sample_length
	for clip in range(num_clips):
		audio_clip = audio[:, :, slice(start * audio_offset, end * audio_offset)].to(device)
		video_clip = video[:, slice(start, end)].to(device)

		# The end of the clip -> throw it away
		if audio_clip.shape[2] < audio_sample_length:
			continue

		audio_input[clip] = audio_clip
		video_input[clip] = video_clip
		start = end
		end += sample_length

	return audio_input, video_input

# TODO: bug with proportion code that eats up memory. Not releasing graph maybe?


def test(args, model, device, valLoader, classes, explanation_mode="test", mean=[0, 0, 0], std=[1, 1, 1]):
	"""
	Args:


	TODO: explanation_mode: {test|simulation}
		This parameter defines the mode in which explanations are generated;
		"test" emulates the way the network was trained and is matched to the
		test level performance.
		"simulation" emulates ad-hoc processing of the data with a "live"
		explanation, and therefore takes up more processing and data.

	Create full videos of frames / samples and use the video-level average.
	Also log the explanation results up to weights and biases.
	"""

	model.eval()
	sample_length = 16
	audio_sample_length = 96
	correct = 0

	def register_hooks(layer, in_, grad_):
		def get_in(self, l_in, l_out):
			in_.append(l_in[0])

		def get_grad(self, grad_in, grad_out):
			grad_.append(grad_in[0])
		fwd = layer.register_forward_hook(get_in)
		bkd = layer.register_backward_hook(get_grad)
		return fwd, bkd

	def remove_hooks(hooks):
		for i in hooks:
			i.remove()

	for idx, (audio, video, type, target, clip_id) in tqdm(enumerate(valLoader, 0), total=len(valLoader.dataset)):

		# Get single item from batch, batch size cannot be more than 1 at the moment
		audio = audio[0, :, :, :]
		video = video[0, :, :, :, :]
		type = type[0]
		target = target[0]
		clip_id = clip_id[0]

		audio_input, video_input = _construct_test_batch(audio, video, sample_length, audio_sample_length)
		audio_input, video_input = audio_input.to(device), video_input.to(device)
		num_clips = video_input.shape[0]

		"""
		# change video to only a single frame
		for i in range(video_input.size(0)):
			for j in range(video_input[i].size(1)):
				video_input[i, :, j, :, :] = video_input[0, :, 10, :, :]
		"""

		"""
		# remove every other frame
		for i in range(video_input.size(0)):
			for j in range(video_input[i].size(1)):
				if j % 2 == 0:
					video_input[i, :, j, :, :] = video_input[i, :, j - 1, :, :]
		"""

		# Register hooks for relevance proportion
		gradients, inputs = [], []
		hooks = register_hooks(model.fc1, inputs, gradients)

		# Pass the video as a batch to the network FIXME: (could be sped up if we use batches)
		output = model(video_input, audio_input)

		# -*- Model performance -*-
		#correct += _per_video_performance(args, model, device, output, target)

		pred = output.max(1, keepdim=True)[1]
		per_video_pred = torch.mode(pred.squeeze().float())[0]
		correct += per_video_pred.eq(target.view_as(per_video_pred))

		# -*- Selective relevance for batch -*-

		# Store the gradient of the output w.r.t the target
		filter_out = torch.zeros_like(output)
		filter_out[:, target] = 1

		# Get the gradient of each input
		video_gradient = torch.autograd.grad(
			output,
			video_input,
			grad_outputs=filter_out,
			retain_graph=True)[0]  # NOTE: why not retain graph here?

		audio_gradient = torch.autograd.grad(
			output,
			audio_input,
			grad_outputs=filter_out,
			retain_graph=True)[0]

		# -*- batch-level discriminative relevance -*-
		proportions = _per_video_proportion_relevance(num_clips, inputs, gradients, num_video_features=1000, num_audio_features=1000)
		audio_ratio, video_ratio = proportions

		target_index = target
		target_index = int(target_index.detach().numpy())
		results_dir = f"./results/{valLoader.dataset.actionClasses[target_index]}/{clip_id}"
		Path(results_dir).mkdir(parents=True, exist_ok=True)

		# -*- save inputs -*-
		#np.save(os.path.join(results_dir,"audio_input"),audio_input.detach().cpu().numpy().transpose(0,2,3,1))
		#np.save(os.path.join(results_dir,"video_input"),video_input.detach().cpu().numpy().transpose(0,2,3,4,1))

		#wandb.save(results_dir + "/audio_input.npy")
		#wandb.save(results_dir + "/video_input.npy")

		# -*- save prediction
		Path(f"{results_dir}/pred_{valLoader.dataset.actionClasses[int(per_video_pred.detach().cpu().numpy())]}").touch()

		# save proportions to file
		Path(f"{results_dir}/audio_{audio_ratio:.3f}-video_{video_ratio:.3f}.ratio").touch()

		clip_audio_spectral_relevance = []
		clip_audio_temporal_relevance = []
		clip_video_spatial_relevance = []
		clip_video_temporal_relevance = []

		for clip in range(num_clips):

			# TODO: clip level proportion of relevance
			video_clip = video_input[clip].detach().cpu().numpy()
			audio_clip = audio_input[clip].detach().cpu().numpy()

			video_relevance = video_gradient[clip].sum(dim=0)
			audio_relevance = audio_gradient[clip][0]

			sr = _clip_level_selective_relevance(
				video_clip,
				video_relevance,
				audio_clip,
				audio_relevance,
				mean=mean,
				std=std)

			clip_audio_spectral_relevance.append(sr[3])
			clip_audio_temporal_relevance.append(sr[2])
			clip_video_spatial_relevance.append(sr[1])
			clip_video_temporal_relevance.append(sr[0])

		#np.save(os.path.join(results_dir, "audio_spectral_relevance"), np.asarray(clip_audio_spectral_relevance))
		#np.save(os.path.join(results_dir, "audio_temporal_relevance"), np.asarray(clip_audio_temporal_relevance))
		#np.save(os.path.join(results_dir, "video_spatial_relevance"), np.asarray(clip_video_spatial_relevance))
		#np.save(os.path.join(results_dir, "video_temporal_relevance"), np.asarray(clip_video_temporal_relevance))

		#wandb.save(os.path.join(results_dir, "audio_spectral_relevance.npy"))
		#wandb.save(os.path.join(results_dir, "audio_temporal_relevance.npy"))
		#wandb.save(os.path.join(results_dir, "video_spatial_relevance.npy"))
		#wandb.save(os.path.join(results_dir, "video_temporal_relevance.npy"))

		remove_hooks(hooks)
	accuracy = 100. * correct / len(valLoader.dataset)
	return accuracy, audio_gradient, video_gradient


def post_train_explain(modelPath, dataLocation, classesLocation, config):
	use_cuda = not config['no_cuda'] and torch.cuda.is_available()
	device = torch.device('cuda:0' if (torch.cuda.is_available() and not config['no_cuda']) else 'cpu')
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# transfors for dataset
	train_transforms, val_transforms, test_transforms = getTransforms()
	mean, std, norm_value = getMeanStdNorm()

	valLoader = torch.utils.data.DataLoader(
		dataset.AudioVideoDataset(root=dataLocation + "/test", classesPath=classesLocation, **test_transforms),
		#dataset.AudioVideoDataset(root=dataLocation + "/valss", classesPath=classesLocation, **test_transforms),
		#dataset.AudioVideoDataset(root=dataLocation + "/fyp", classesPath=classesLocation, **test_transforms),
		batch_size=1, shuffle=False, **kwargs)
	classes = valLoader.dataset.actionClasses
	valLoader.dataset.setStage(2)  # we only want positive examples

	# -*- load in best val accuracy and explain everything in test set -*-
	preTrainedModel = torch.load(modelPath, map_location=torch.device('cpu'))
	model = mobilenets.FusionNet(train=False)
	model.load_state_dict(preTrainedModel['state_dict'])
	model.to(device)

	test(config, model, device, valLoader, classes, mean=mean, std=std)


if __name__ == "__main__":
	hyperparameterDefaults = {
		"no_cuda": False,
		"seed": 42
	}

	wandb.init(project="fusion-reproduction", config=hyperparameterDefaults)
	config = wandb.config

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='Path of saved weights')
	args = parser.parse_args()

	modelPath = args.model
	dataLocation = "./data/fusion_data/UCF-101"
	classesLocation = "classIndUcf101.txt"
	post_train_explain(modelPath, dataLocation, classesLocation, hyperparameterDefaults)
