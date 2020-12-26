import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from actionTraining import actionTrain, printAccuracy
from preTraining import pretrain
import explain
import models.mobilenet_v1 as mobilenets
from losses import ContrastiveLoss
import argparse
import time

from torchsummary import summary


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 1
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
	return nn.Sequential(*layers)


if __name__ == "__main__":
	# Learning rate is pass in as an argument
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning_rate", help="Learning rate for wandb sweeps",
						type=float)
	parser.add_argument("--action", choices=['pre', 'action', 'val', 'audio', 'video'], help="Type of run")
	parser.add_argument('--dataset', choices=['kinetics600', 'UCF-101'], default='UCF-101',
						help='Dataset to use (default: UCF-101)')
	parser.add_argument('--batch', type=int, default=16,
						help='Batch size (default: 16)')
	parser.add_argument('--model', help='Path of saved weights')
	args = parser.parse_args()

	torch.manual_seed(42)  # reproducibility

	hyperparameterDefaults = {
		"batch_size": args.batch,
		"val_batch_size": 64,
		"epochs": 140,
		"lr": args.learning_rate,
		"lr_steps": [55, 80, 90, 100, 110, 120, 130],
		"momentum": 0.9,
		"dampening": 0.9,
		"weight_decay": 1e-3,
		"nesterov": False,
		"no_cuda": False,
		"log_interval": 10,
		"seed": 42,
		"dataset": args.dataset
	}

	# We only need to log if running training
	if args.action != "val":
		wandb.init(project="fusion-reproduction", config=hyperparameterDefaults)
		config = wandb.config

	use_cuda = not hyperparameterDefaults['no_cuda'] and torch.cuda.is_available()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

	print("Training on", device)

	# configure dataset to use
	ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
	if args.dataset == 'UCF-101':
		dataLocation = ROOT_DIR + "/data/fusion_data/UCF-101"
		classesLocation = "classIndUcf101.txt"
	else:
		dataLocation = ROOT_DIR + "/data/fusion_data/Kinetics600"
		classesLocation = "classIndKinetics.txt"

	# pre training or action classification training
	if args.action == "pre":
		videoNet = mobilenets.VideoMobileNet()
		audioNet = mobilenets.AudioMobileNet(mel=False, device=device)
		model = mobilenets.SyncNet(videoNet, audioNet)

		# If possible traing on multiple GPUs
		#if use_cuda and torch.cuda.device_count() > 1:
		#    print("Training on", torch.cuda.device_count(), "GPUs!")
		#    model = nn.DataParallel(model)
		#elif torch.cuda.is_available() == False:
		#    print("Training on CPU!")

		model.to(device)

		#for name, param in model.named_parameters():
		#    print(name, param.requires_grad)
		#    print(name, torch.mean(param))

		pretrain(model, config, dataLocation, classesLocation, kwargs, device)
	if args.action == "audio":
		audioNet = mobilenets.AudioClassifier()

		audioNet.to(device)

		actionTrain(audioNet, config, dataLocation, classesLocation, kwargs, device, modality="audio")
	if args.action == "video":
		videoNet = mobilenets.VideoClassifier()

		videoNet.to(device)

		actionTrain(videoNet, config, dataLocation, classesLocation, kwargs, device, modality="video")
	elif args.action == "action":
		# If you need to create the sub net saves then uncomment this first
		"""
		videoNet = mobilenets.VideoMobileNet()
		audioNet = mobilenets.AudioMobileNet(mel=False)
		model = mobilenets.SyncNet(videoNet, audioNet)
		checkpoint = torch.load("./saves/epoch_99_save.pth", map_location=torch.device('cpu'))

		model.load_state_dict(checkpoint['state_dict'])
		model.to(device)

		torch.save({
			'state_dict': model.videoNet.state_dict()
		}, './saves/videoNetBest.pth')

		torch.save({
			'state_dict': model.audioNet.state_dict()
		}, './saves/audioNetBest.pth')

		print("done seperation of subnets")
		"""

		# End of saving sub nets

		# Create model for action recognition from pre trained weights
		#videoPath = "./saves/videoNetBest.pth"
		videoPath = "./saves/trainingAction/video_save.pth"
		#videoPath = "./saves/video_9_save.pth"
		# videoPath = "./saves/kinetics_mobilenet_save.pth"
		#audioPath = "./saves/audioNetBest.pth"
		audioPath = "./saves/trainingAction/audio_save.pth"


		#fusionNet = mobilenets.FusionNetLoaded(videoPath, audioPath, videoEmbedding=1000, mel=False)

		fusionNet = mobilenets.FusionNet()

		fusionNet.to(device)

		#print(fusionNet)

		#summary(fusionNet, [(3, 16, 122, 122), (1, 64, 98)])

		#for name, param in fusionNet.named_parameters():
		#    print(name, param.requires_grad)
		#for name, param in fusionNet.videoModel.named_parameters():
		#    print(name, torch.mean(param))

		#for name, param in fusionNet.videoModel.named_parameters():
		#for name, param in fusionNet.audioModel.named_parameters():
		#for name, param in audioModel.named_parameters():
			#print(name, torch.mean(param))
			#print(name, param)

		actionTrain(fusionNet, config, dataLocation, classesLocation, kwargs, device, modality="both")
	# This is for validation, it does not perform training
	elif args.action == "val":
		modelPath = args.model

		preTrainedModel = torch.load(modelPath, map_location=torch.device('cpu'))

		fusionNet = mobilenets.FusionNet()

		fusionNet.load_state_dict(preTrainedModel['state_dict'])

		printAccuracy(fusionNet, hyperparameterDefaults['val_batch_size'], dataLocation, classesLocation, kwargs, device)
