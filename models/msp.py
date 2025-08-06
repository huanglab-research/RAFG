import torch
import torch.nn as nn
import random


def generate_2d_mask(H=16, W=8, left=0, top=0, width=8, height=8, part=-1, cls_label=True, device='cuda'):
	H, W, left, top, width, height = \
	int(H), int(W), int(left), int(top), int(width), int(height)
	assert left + width <= W and top + height <= H
	l, w = sorted(random.sample(range(left, left + width + 1), 2))
	t, h = sorted(random.sample(range(top, top + height + 1), 2))
	# l,w,t,h = left, left+width, top, top+height ### for test
	mask = torch.zeros([H, W], device=device)
	mask[t : h + 1, l : w + 1] = 1
	mask = mask.flatten(0)
	mask_ = torch.zeros([len(mask) + 4], device=device)
	mask_[4:] = mask
	mask_[part] = 1
	mask_[0] = 1 if cls_label else 0 ######### cls token
	mask_ = mask_.unsqueeze(1) # N x 1
	mask_ = mask_ @ mask_.t() # N x N
	return mask_

def attn_mask_generate(self, N=132, H=16, W=8, device='cuda'):
	mask = torch.ones(N,1, device=device)
	mask[1 : 4, 0] = 0
	mask_ = (mask @ mask.t()).bool()
	mask_ |= generate_2d_mask(H,W,0,0,W,H/2,1,False, device).bool()
	mask_ |= generate_2d_mask(H,W,0,H/4,W,H/2,2, False, device).bool()
	mask_ |= generate_2d_mask(H,W,0,H/2,W,H/2,3, False, device).bool()
	mask_[1 : 4, 0] = True
	mask_[0, 1 : 4] = True
	return mask_


generate_2d_mask()