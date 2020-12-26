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
import argparse  # Only used for wandb sweeps

import matplotlib.pyplot as plt


def train(args, model, device, trainLoader, optimizer, criterion, modality):
	"""Training for action recognition"""
	model.train()  # switch to train mode
	losses = []

	for i, (audio, video, type, target, clip_id) in enumerate(trainLoader, 0):

		# zero the parameter gradients
		optimizer.zero_grad()

		#print(audio.min())
		#print(audio.max())
		#print(audio.mean())
		#print("===")

		"""
		for i in range(audio.size(0)):
			print(audio[i, 0, :, :].size())
			print(audio[i, 0, :, :])
			plt.imshow(audio[i, 0, :, :])
			plt.show()
		"""

		audio, video, target = audio.to(device), video.to(device), target.to(device)

		if modality == "both":
			output = model(video, audio)
		elif modality == "audio":
			output = model(audio)
		else:
			output = model(video)

		loss = criterion(output, target)

		#print(target)
		#print(torch.argmax(output, dim=1))
		#print("========")

		loss.backward()
		losses.append(loss.item())

		optimizer.step()
	averageLoss = sum(losses) / len(losses)
	wandb.log({"Train loss": averageLoss})
	return averageLoss


def test(config, model, device, valLoader, criterion, modality):
	"""Testing for action recognition"""
	model.eval()  # switch to evaluate mode
	losses = []
	correct = 0
	total = 0

	with torch.no_grad():
		for i, (audio, video, type, target, clip_id) in enumerate(valLoader, 0):

			audio, video, target = audio.to(device), video.to(device), target.to(device)

			if modality == "both":
				output = model(video, audio)
			elif modality == "audio":
				output = model(audio)
			else:
				output = model(video)

			loss = criterion(output, target)
			losses.append(loss.item())

			# Get the index of the max log-probability
			pred = output.max(1, keepdim=True)[1]
			total += pred.eq(target.view_as(pred)).sum().item()

	# Weights & Biases logging
	accuracy = 100. * total / len(valLoader.dataset)
	averageLoss = sum(losses) / len(losses)
	wandb.log({
		"Val Accuracy": accuracy,
		"Val Loss": averageLoss})

	return accuracy


def printAccuracy(model, batchSize, dataLocation, classesLocation, kwargs, device):
	"""Testing for action recognition"""
	# transfors for dataloader
	train_transforms, val_transforms, test_transforms = getTransforms()

	# data loader
	testLoader = torch.utils.data.DataLoader(
		dataset.AudioVideoDataset(root=dataLocation + "/test", classesPath=classesLocation, **val_transforms),
		batch_size=batchSize, shuffle=False, **kwargs)

	model.to(device)

	testLoader.dataset.setStage(2)  # we only want positives for validation

	model.eval()  # switch to evaluate mode
	class_correct = list(0. for i in range(testLoader.dataset.numClasses))
	class_total = list(0. for i in range(testLoader.dataset.numClasses))

	with torch.no_grad():
		for i, (audio, video, type, target, clip_id) in enumerate(testLoader, 0):

			audio, video, target = audio.to(device), video.to(device), target.to(device)

			"""
			for i in range(video.size(0)):
				for j in range(video[i].size(1)):
					if j % 2 == 0:
						video[i, :, j, :, :] = video[i, :, j - 1, :, :]

			# change video to only a single frame
			for i in range(video.size(0)):
				for j in range(video[i].size(1)):
					video[i, :, j, :, :] = video[i, :, 10, :, :]
			"""

			output = model(video, audio)

			# Get the index of the max log-probability
			pred = output.max(1, keepdim=True)[1]

			for i in range(len(target)):
				#print(target[i])
				label = target[i]
				if pred[i].eq(label.view_as(pred[i])):
					class_correct[label] += 1
				class_total[label] += 1

		for i in range(testLoader.dataset.numClasses):
			print('Accuracy of %5s : %2d %%' % (
				testLoader.dataset.actionClasses[i], 100 * class_correct[i] / class_total[i]))

	# Weights & Biases logging
	accuracy = 100. * sum(class_correct) / sum(class_total)
	print("Total accuracy:", accuracy)


def actionTrain(model, config, dataLocation, classesLocation, kwargs, device, modality="both"):
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True

	# transfors for dataloader
	train_transforms, val_transforms, test_transforms = getTransforms()

	# comment out if audio transform handled outside of model
	#del train_transforms['audio_transform']
	#del val_transforms['audio_transform']

	# data loaders
	trainLoader = torch.utils.data.DataLoader(
		dataset.AudioVideoDataset(root=dataLocation + "/train", classesPath=classesLocation, **train_transforms),
		batch_size=config.batch_size, shuffle=True, **kwargs)

	valLoader = torch.utils.data.DataLoader(
		dataset.AudioVideoDataset(root=dataLocation + "/val", classesPath=classesLocation, **val_transforms),
		batch_size=config.val_batch_size, shuffle=False, **kwargs)

	optimizer = optim.SGD(
		model.parameters(),
		lr=config.lr,
		momentum=config.momentum,
		dampening=config.dampening,
		weight_decay=config.weight_decay,
		nesterov=config.nesterov
	)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, 'min', patience=10)

	criterion = nn.CrossEntropyLoss()
	if device != 'cpu':
		criterion = criterion.to(device)

	model.to(device)

	# WandB - watch
	wandb.watch(model, log="all")

	trainLoader.dataset.setStage(2)  # we only want positives for this training
	valLoader.dataset.setStage(2)  # we only want positives for this training

	print("Starting training at:", time.ctime(time.time()))
	best_accuracy = 0
	lr = 0
	for epoch in tqdm(range(1, config.epochs + 1)):

		adjust_learning_rate(optimizer, epoch, config)

		# get current learning rate
		for param_group in optimizer.param_groups:
			lr = param_group['lr']

		wandb.log({
			"learning_rate": lr,
			"epoch": epoch}
		)

		train(config, model, device, trainLoader, optimizer, criterion, modality=modality)
		newAccuracy = test(config, model, device, valLoader, criterion, modality=modality)

		if newAccuracy > best_accuracy:
			torch.save({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()
			}, './saves/trainingAction/%s_save.pth' % (modality))

			wandb.save('./saves/trainingAction/%s_save.pth' % (modality))
			best_accuracy = newAccuracy


if __name__ == "__main__":
	# Learning rate is pass in as an argument
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--learning_rate",
		help="Learning rate for wandb sweeps",
		type=float)
	args = parser.parse_args()

	torch.manual_seed(0)  # reproducibility

	hyperparameter_defaults = dict(
		batch_size=12,
		start_lr=args.learning_rate,
		epochs=90,
	)

	wandb.init(project="fusion-reproduction", config=hyperparameter_defaults)
	config = wandb.config

	# Load and confugure Weights and Biases
	wandb.save(os.path.join("./saves/trainingAction/*.pth"))  # backup to wandb
	wandb.save(os.path.join("./saves/completeAction/*.pth"))  # backup to wandb

	dataloader = FusionDataloader("./data/UCF-101")
	d = dataloader.fusionDataset

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	videoNet = mobilenets.VideoMobileNet()
	audioNet = mobilenets.AudioMobileNet(device=device)
	fusionNet = mobilenets.FusionNet(videoNet, audioNet)
	checkpoint = torch.load("./saves/training/epoch_0_AVTS_save.pth", map_location=device)
	fusionNet.load_state_dict(checkpoint['model_state_dict'], strict=False)
	fusionNet.eval()
	optimizer = optim.SGD(fusionNet.parameters(), lr=args.learning_rate)
	#optimizer = optim.SGD(SyncNet.parameters(), lr=hyperparameter_defaults['start_lr'])

	# If possible train on GPU
	fusionNet.to(device)

	# torch.save(SyncNet, './saves/completeAction/trained_AVTS.pth')
