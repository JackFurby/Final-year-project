import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models.mobilenet_v1 as mobilenets
import dataset
import wandb
from losses import ContrastiveLoss
import time
import os
import random
from spatial_transforms import *
from temporal_transforms import *
from transforms import *
from utils import *
from tqdm import tqdm
import argparse  # Only used for wandb sweeps


def train(args, model, device, trainLoader, optimizer, criterion):
	"""Training for AVTS"""
	model.train()  # switch to train mode
	losses = []

	for i, (audio, video, type, target, clip_id) in enumerate(trainLoader, 0):

		# zero the parameter gradients
		optimizer.zero_grad()

		audio, video = audio.to(device), video.to(device)

		videoOut, audioOut = model(video, audio)

		yTrue = []
		for j in range(video.size(0)):
			if type[j] == "positive":
				yTrue.append(1)
			else:
				yTrue.append(0)

		yTrue = torch.tensor(yTrue).to(device)

		loss = criterion(videoOut, audioOut, yTrue)

		loss.backward()
		losses.append(loss.item())

		optimizer.step()
	averageLoss = sum(losses) / len(losses)
	wandb.log({"Train loss": averageLoss})
	return averageLoss


def test(args, model, device, valLoader, criterion):
	"""Testing for AVTS"""
	model.eval()  # switch to evaluate mode
	losses = []

	with torch.no_grad():
		for i, (audio, video, type, target, clip_id) in enumerate(valLoader):

			audio, video = audio.to(device), video.to(device)

			videoOut, audioOut = model(video, audio)

			yTrue = []
			for j in range(video.size(0)):
				if type[j] == "positive":
					yTrue.append(1)
				else:
					yTrue.append(0)

			yTrue = torch.tensor(yTrue).to(device)

			loss = criterion(videoOut, audioOut, yTrue)

			losses.append(loss.item())

	averageLoss = sum(losses) / len(losses)
	wandb.log({"Test loss": averageLoss})
	return averageLoss


# Update the learning rate by 0.1
def updateLearningRate(optimizer):
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * 0.1

		return optimizer


def runEpoch(config, model, device, trainLoader, valLoader, optimizer, criterion, epoch, best_loss, lastLossChange, bestTrainLoss):
	wandb.log({"epoch": epoch})  # add epoch to w and b

	# Log learning rate
	for param_group in optimizer.param_groups:
		wandb.log({"learning_rate": param_group['lr']})

	newTrainLoss = train(config, model, device, trainLoader, optimizer, criterion)
	newLoss = test(config, model, device, valLoader, criterion)

	# Update the learning rate if loss does not decrease for 5 epochs
	if (bestTrainLoss is -1) or (newTrainLoss < bestTrainLoss):
		bestTrainLoss = newTrainLoss
		lastLossChange = 0
	elif lastLossChange >= 5:
		optimizer = updateLearningRate(optimizer)
		lastLossChange = 0
	elif newTrainLoss >= bestTrainLoss:
		lastLossChange += 1

	if (newLoss < best_loss) or (best_loss is -1):
		torch.save({
			'epoch': epoch,
			'state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()
		}, './saves/epoch_%d_save.pth' % (epoch))

		wandb.save('./saves/epoch_%d_save.pth' % (epoch))

		torch.save({
			'epoch': epoch,
			'state_dict': model.videoNet.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()
		}, './saves/videoNetBest.pth')

		wandb.save('./saves/videoNetBest.pth')

		torch.save({
			'epoch': epoch,
			'state_dict': model.audioNet.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()
		}, './saves/audioNetBest.pth')

		wandb.save('./saves/audioNetBest.pth')
		best_loss = newLoss

	return best_loss, lastLossChange, bestTrainLoss


def pretrain(model, config, dataLocation, classesLocation, kwargs, device, startEpoch=1, optimizer=None):
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

	# If optimizer is not provided, make one
	if optimizer == None:
		optimizer = optim.SGD(
			model.parameters(),
			lr=config.lr,
			momentum=config.momentum,
			dampening=config.dampening,
			weight_decay=config.weight_decay,
			nesterov=config.nesterov
		)

	criterion = ContrastiveLoss(0.99)

	wandb.watch(model, log="all")

	print("Started training at:", time.ctime(time.time()))
	best_loss = -1
	bestTrainLoss = -1
	lastLossChange = 0

	if startEpoch >= 50:
		trainLoader.dataset.setStage(1)
		valLoader.dataset.setStage(1)

	for epoch in tqdm(range(startEpoch, config.epochs + 1)):
		best_loss, lastLossChange, bestTrainLoss = runEpoch(
			config,
			model,
			device,
			trainLoader,
			valLoader,
			optimizer,
			criterion,
			epoch,
			best_loss,
			lastLossChange,
			bestTrainLoss
		)
		# Move onto harder examples at epoch 51 (start epoch is 1)
		if epoch == 51:
			trainLoader.dataset.setStage(1)
			valLoader.dataset.setStage(1)
			print("Training on harder examples at:", time.ctime(time.time()))


if __name__ == "__main__":
	"""Resuming from a checkpoint"""

	torch.manual_seed(40)  # reproducibility

	hyperparameterDefaults = {
		"batch_size": 32,
		"val_batch_size": 64,
		"epochs": 140,
		"lr": 0.1,
		"lr_steps": [55, 80, 90, 100, 110, 120, 130],
		"momentum": 0.9,
		"dampening": 0.9,
		"weight_decay": 1e-3,
		"nesterov": False,
		"no_cuda": False,
		"log_interval": 10,
		"seed": 40,
		"dataset": "kinetics600"
	}

	wandb.init(project="fusion-reproduction", config=hyperparameterDefaults)
	config = wandb.config

	use_cuda = not hyperparameterDefaults['no_cuda'] and torch.cuda.is_available()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

	# configure dataset to use
	dataLocation = "./data/fusion_data/UCF-101"
	classesLocation = "classIndUcf101.txt"
	#dataLocation = "./data/fusion_data/Kinetics600"
	#classesLocation = "classIndKinetics.txt"

	checkpoint = torch.load("./saves/epoch_5_save.pth", map_location=torch.device('cpu'))

	videoNet = mobilenets.VideoMobileNet()
	audioNet = mobilenets.AudioMobileNet(mel=True, device=device)
	model = mobilenets.SyncNet(videoNet, audioNet)

	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)

	optimizer = optim.SGD(
		model.parameters(),
		lr=config.lr,
		momentum=config.momentum,
		dampening=config.dampening,
		weight_decay=config.weight_decay,
		nesterov=config.nesterov)
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	startEpoch = checkpoint['epoch'] + 1

	pretrainResumed(model, config, dataLocation, classesLocation, kwargs, device, startEpoch, optimizer)
