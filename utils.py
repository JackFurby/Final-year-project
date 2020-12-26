import shutil
import numpy as np
import logging

import torch
import torch.nn as nn
from count_hooks import *


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def calculate_accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def save_checkpoint(state, is_best, opt):
	torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
	if is_best:
		shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best.pth' % (opt.result_path, opt.store_name))


def adjust_learning_rate(optimizer, epoch, opt):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr_new = opt['lr'] * (0.1 ** (sum(epoch >= np.array(opt['lr_steps']))))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr_new


register_hooks = {
	nn.Conv2d: count_conv2d,
	nn.Conv3d: count_conv3d,
	nn.BatchNorm2d: count_bn2d,
	nn.BatchNorm3d: count_bn2d,
	nn.ReLU: count_relu,
	nn.ReLU6: count_relu,
	nn.MaxPool1d: count_maxpool,
	nn.MaxPool2d: count_maxpool,
	nn.MaxPool3d: count_maxpool,
	nn.AvgPool1d: count_avgpool,
	nn.AvgPool2d: count_avgpool,
	nn.AvgPool3d: count_avgpool,
	nn.Linear: count_linear,
	nn.Dropout: None,
}


def profile(model, xa_size, xb_size=None, custom_ops={}):
	def add_hooks(m):
		if len(list(m.children())) > 0:
			return

		m.register_buffer('total_ops', torch.zeros(1))
		m.register_buffer('total_params', torch.zeros(1))

		for p in m.parameters():
			m.total_params += torch.Tensor([p.numel()])

		m_type = type(m)
		fn = None

		if m_type in custom_ops:
			fn = custom_ops[m_type]
		elif m_type in register_hooks:
			fn = register_hooks[m_type]
		else:
			logging.warning("Not implemented for ", m)

		if fn is not None:
			logging.info("Register FLOP counter for module %s" % str(m))
			m.register_forward_hook(fn)

	model.eval()
	model.apply(add_hooks)

	xa = torch.zeros(xa_size)
	if xb_size:
		xb = torch.zeros(xb_size)
		model(xa,xb)
	else:
		model(xa)

	total_ops = 0
	total_params = 0
	for m in model.modules():
		if len(list(m.children())) > 0: # skip for non-leaf module
			continue
		total_ops += m.total_ops
		total_params += m.total_params

	total_ops = total_ops.item()
	total_params = total_params.item()

	return total_ops, total_params
