import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
	"""
	Contrastive loss

	distance-based loss function which takes the individual predictions of audio
	and video for the distance calculation
	"""

	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin  # Default is 2

	def forward(self, audio, video, yTrue):
		"""
		Args:
			yTrue should be given the value of 0 if the audio and video segments
			are from a negative example, otherwise it should have the value of 1
		"""
		# euclidean distance
		distance = F.pairwise_distance(video, audio, keepdim=True)

		losses = yTrue * distance.pow(2) + (1 - yTrue) * F.relu(self.margin - distance, 0).pow(2)
		# losses = yTrue * torch.pow(distance, 2) + (1 - yTrue) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

		return torch.mean(losses)
