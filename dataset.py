import glob
import os
import soundfile as sf
from os import path
from PIL import Image
import numpy as np
from numpy.random import randint
import torch.utils.data as data
import random
from tqdm import tqdm
import cv2
import torch

# TODO: abstract AVTS and FP out from FusionDataset base
# TODO: inherit kinetics dataset and ucf-101 dataset
# TODO: ucf-101 - if no audio, cause no input. How?
"""
- Both datasets load the entire video
- AVTS applies logic for selecting hard/soft negs
- FP segments the whole video (like TSN)
"""


class AudioVideoDataset(data.Dataset):
	def __init__(
		self,
		root,
		classesPath,
		sample_duration=16,
		spatial_transform=None,
		temporal_transform=None,
		audio_transform=None):

		self.setStage(0)  # initial stage

		self.data = []
		self.sample_duration = sample_duration
		self.spatial_transform = spatial_transform
		self.temporal_transform = temporal_transform
		self.audio_transform = audio_transform

		self.audio_list, self.video_list, self.labels = [], [], []
		audio_paths = glob.glob(path.join(root + '/**/audio/*.wav'), recursive=True)

		# Only include data that has an audio track
		for i in tqdm(range(len(audio_paths))):
			label, _, clip = audio_paths[i].split("/")[-3:]
			video = path.join(root, label, "frames", clip[:-4])
			number_frames = len(os.listdir(video))
			frame_indices = list(range(number_frames))

			# if some frame counts are not 16 this will filter them out
			if number_frames >= 16:
				data = {
					'video': video,
					'audio': audio_paths[i],
					'frame_indices': frame_indices,
					'label': label,
					'clip': clip[:-4]}

				self.data.append(data)

		# Create dict for actions to int mapping
		self.setActionClasses(classesPath)

		self.numClasses = len(self.actionClasses)  # Number of action classes

		# Class to index transform
		for idx, data in enumerate(self.data):
			self.data[idx]['label'] = self.reverseActionClasses.get(self.data[idx]['label'])

	def __len__(self):
	    return len(self.video_list)

	def __getitem__(self, index):
		"""
		Args:
		    index (int): Index
		Returns:
		    tuple: (sample, target) where target is class_index of the target class.
		"""
		# video: 16fps, audio: 16khz
		# 1 video frame == 1000 audio samples
		# 1 segment == 16000 audio samples == 16 frames
		video_path = self.data[index]['video']
		audio_path = self.data[index]['audio']
		frame_indices = self.data[index]['frame_indices']
		clip_id = self.data[index]['clip']

		# crop video to length
		if self.temporal_transform is not None:
			frame_indices = self.temporal_transform(frame_indices)

		# 50% of samples are positive, 50% are negative
		# if stage is 2 all are positive
		if random.uniform(0, 1) <= 0.5 or self.stage is 2:
			audio_clip = self._get_audio(audio_path, frame_indices)
			type = 'positive'
		else:
			# Only make hard examples is stage is 1
			# 75% of negatives are easy, 25% are hard
			# gap of at least 0.5 seconds (assuming 1 sec video)
			if random.uniform(0, 1) <= 0.25 and self.stage is 1:
				dur = self.sample_duration
				number_frames = len(os.listdir(video_path))
				all_frame_indices = list(range(number_frames))

				# Work out if there is more audio before or after frames
				lenBefore = len(all_frame_indices[all_frame_indices[0]:frame_indices[0]])
				lenAfter = len(all_frame_indices[frame_indices[-1]:all_frame_indices[-1]])
				if (lenBefore > lenAfter):
					before = True
				else:
					before = False

				# More audio before frames and space to take a sample
				if before and ((frame_indices[0] - (dur / 2) - dur) > 0):
					audioStart = random.randint(0, (frame_indices[0] - (dur / 2) - dur))
				# More audio after frames / not enough before frames
				else:
					# if not enough space to get a sample start from last indice
					if (frame_indices[-1] + dur + (dur / 2)) >= all_frame_indices[-1]:
						audioStart = all_frame_indices[-1]
					else:
						audioStart = random.randint((frame_indices[-1] + (dur / 2)), all_frame_indices[-1] - dur)
				audioEnd = audioStart + dur
				newAudioIndices = list(range(audioStart, audioEnd))

				audio_clip = self._get_audio(audio_path, newAudioIndices)
				type = 'hard'
			else:
				# get audio sample from differnt video
				newIndex = random.randint(0, len(self.data) - 1)
				audio_path = self.data[newIndex]['audio']
				newAudioIndices = list(range(0, self.sample_duration - 1))  # as we dont know lengh, get audio from start of video
				audio_clip = self._get_audio(audio_path, newAudioIndices)
				type = 'easy'

		# Audio transform
		if self.audio_transform is not None:
			audio_clip = self.audio_transform(audio_clip)
		else:
			audio_clip = torch.from_numpy(audio_clip).float()

		# -*- Video loading -*- #
		video_clip = self._get_frames(video_path, frame_indices)
		if self.spatial_transform is not None:
			self.spatial_transform.randomize_parameters()
			video_clip = [self.spatial_transform(img) for img in video_clip]
		video_clip = torch.stack(video_clip, 0).permute(1, 0, 2, 3)

		# Get target
		target = self.data[index]['label']

		return audio_clip, video_clip, type, target, clip_id

	def _get_frames(self, video_dir_path, frame_indices):
		cap = cv2.VideoCapture(f"{video_dir_path}/%04d.jpeg")
		video = []
		cap.set(1, frame_indices[0])
		for _ in frame_indices:
			ret, frame = cap.read()
			if ret:
				pil_frame = Image.fromarray(frame)
				video.append(pil_frame)
			else:
				break

		cap.release()

		# For videos less than 1s long, loop around. Same for audio above.
		for frame in video:
			if len(video) >= self.sample_duration:
				break
			video.append(frame)

		if len(video) == 0:  # give an empty clip # FIXME: is this right?
			for _ in range(self.sample_duration):
				video.append(Image.new('RGB', (320, 180)))
		return video

	def __len__(self):
		return len(self.data)

	def _get_audio(self, audio_clip_path, frame_indices):
		wav_data, fs = sf.read(audio_clip_path)
		assert fs == 16000
		if len(wav_data.shape) > 1:
			wav_data = np.mean(wav_data, axis=1)
		#wav_data = wav_data / 32768.0  # values between -1 and +1

		audio_sample_duration = self.sample_duration * 1000

		indices = slice(
			frame_indices[0] * 1000,
			(frame_indices[-1] + 1) * 1000
		)

		audio = wav_data[indices]

		# For videos that are less than 1s long,
		# pad the ending with nothing (zeros)
		if len(audio) < audio_sample_duration:
			pad = np.zeros(audio_sample_duration - len(audio))
			audio = np.append(audio, pad)

		return audio

	# https://discuss.pytorch.org/t/changing-transformation-applied-to-data-during-training/15671
	def setStage(self, stage):
		"""Update stage for curriculum learning

			if stage is 0 then only easy negatives are created
			if stage is 1 then both hard and easy negatives are created
		"""
		self.stage = stage

	def setActionClasses(self, path):
		"""Given a path to a txt file this will return a dict of
		classifications string to int.

		The txt file should be formatted with each classification on a new line
		as follows:
		class_int classification_text

		e.g.
		1 ApplyEyeMakeup
		"""
		d = {}
		rd = {}
		with open(path) as f:
			for line in f:
				line = line.split()
				#action = "".join([x.capitalize() for x in line[1:]])
				# Last line is blank. This will stop error
				# Index in file starts from 1, we need it from 0
				if len(line) >= 2:
					d[int(line[0]) - 1] = " ".join(line[1:])
					rd[" ".join(line[1:])] = int(line[0]) - 1
		self.actionClasses = d
		self.reverseActionClasses = rd


if __name__ == "__main__":
	d = FusionDataset(root="./data/UCF-101")
