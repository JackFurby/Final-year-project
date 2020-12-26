import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torchexplain


def selective_relevance(expl_tensor, sig, cmap="gist_heat", inp=None, img=False):
	"""
	Apply mask to explanation based on it's gradient over time,
	i.e. how quickly relevance changes in a region.

	Args:
		expl_tensor (torch.Tensor): Grey-scale magnitude representation of relevance
									for an input video. Each pixels value should be the
									amount of relevance at that position. Overlaying the
									explanation on the input, or putting it through some
									colour map will cause incorrect results.
		sig: number of standard deviations to separate the dimensions
		cmap (str, optional): Name of matplotlib.pyplot colourmap to apply to the resulting
									Selective Relevance map.
		input ((list of str)/str,optional): The input video on which to overlay the resulting
									Selective Relevance map.
	Returns:
		sel_expl (list of numpy.ndarrays): List of frames for the Selective Relevance map.
	"""
	if isinstance(cmap, str):
		cmap = plt.get_cmap(cmap)

	if expl_tensor.min() < 0:
		expl_tensor = abs(expl_tensor)

	if img:
		x_sobel = torch.tensor(
			[[1, 0, -1],
			[2, 0, -2],
			[1, 0, -1]]).reshape((1 ,1, 3, 3))
		y_sobel = torch.tensor(
			[[1, 2, 1],
			[0, 0, 0],
			[-1,-2,-1]]).reshape((1, 1, 3, 3))
	else:
		sobel = torch.tensor(
			[[[1, 2, 1],
			[2, 4, 2],
			[1, 2, 1]],
			[[0, 0, 0],
			[0, 0, 0],
			[0, 0, 0]],
			[[-1, -2, -1],
			[-2, -4, -2],
			[-1, -2, -1]]]).reshape((1, 1, 3, 3, 3))

	# sobel operator expects a batch and channel dimension,
	# it also requires padding to fit to
	# even dimensions but this can be altered.
	if img:
		deriv_t = F.conv2d(
			expl_tensor[None][None].float(),
			x_sobel.float(),
			padding=(1, 1))[0, 0, ...]
		deriv_s = F.conv2d(
			expl_tensor[None][None].float(),
			y_sobel.float(),
			padding=(1, 1))[0, 0, ...]
	else:
		deriv_t = F.conv3d(
			expl_tensor[None][None].float(),
			sobel.float(),
			padding=(1, 1, 1))[0, 0, ...]

	# this is the selective process in essentially one line:
	# constructing the mask and applying it
	if img:
		temp_vis = (expl_tensor * (abs(deriv_t) > (deriv_t.std() * sig)).float()).numpy()
		spec_vis = (expl_tensor * (abs(deriv_s) > (deriv_s.std() * sig)).float()).numpy()
	else:
		temp_vis = (expl_tensor * (abs(deriv_t) > (deriv_t.std() * sig)).float()).numpy()
		spat_vis = (expl_tensor * (abs(deriv_t) < (deriv_t.std() * sig)).float()).numpy()
	sel_expl = ([],[])
	if img:
		if cmap:
			temp_vis = cmap(temp_vis)
			spec_vis = cmap(spec_vis)
		temp_vis = (temp_vis*255).astype(np.uint8)
		spec_vis = (spec_vis*255).astype(np.uint8)
		temp_vis = cv2.cvtColor(temp_vis,cv2.COLOR_BGRA2BGR)
		spec_vis = cv2.cvtColor(spec_vis,cv2.COLOR_BGRA2BGR)
		if inp is not None:
			in_frame = np.transpose(inp, (1,2,0))
			in_frame = cv2.cvtColor(in_frame*255, cv2.COLOR_GRAY2RGB).astype(np.uint8)
			temp_vis = cv2.addWeighted(temp_vis, 0.5, in_frame, 0.5, 0)
			spec_vis = cv2.addWeighted(spec_vis,0.5,in_frame,0.5,0)
		sel_expl = (temp_vis,spec_vis)
	else:
		for f_idx in range(temp_vis.shape[0]):
			temp_f = temp_vis[f_idx]
			spat_f = spat_vis[f_idx]
			if cmap:
				if isinstance(cmap,tuple):
					temp_f = cmap[0](temp_f)
					spat_f = cmap[1](spat_f)
				else:
					temp_f = cmap(temp_f)
					spat_f = cmap(spat_f)
			temp_f = (temp_f*255).astype(np.uint8)
			spat_f = (spat_f*255).astype(np.uint8)
			temp_f = cv2.cvtColor(temp_f,cv2.COLOR_BGRA2BGR)
			spat_f = cv2.cvtColor(spat_f,cv2.COLOR_BGRA2BGR)
			if inp is not None:
				in_frame = inp[f_idx][...,0:3].astype(np.uint8)
				temp_f = cv2.resize(temp_f,(in_frame.shape[1],in_frame.shape[0]))
				temp_f = cv2.addWeighted(temp_f,0.5,in_frame,0.5,0)
				spat_f = cv2.resize(spat_f,(in_frame.shape[1],in_frame.shape[0]))
				spat_f = cv2.addWeighted(spat_f,0.5,in_frame,0.5,0)
			sel_expl[0].append(temp_f)
			sel_expl[1].append(spat_f)
	return sel_expl


if __name__ == "__main__":
	pass
