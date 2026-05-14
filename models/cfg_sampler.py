# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run
        print('--- creating CFG  1model')
        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        # assert cond_mode in ['only_text', 'only_spatial', 'both_text_spatial','text']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        out = self.model(x, timesteps, y)
        if cond_mode == 'no_cond':
            assert torch.allclose(out, out_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))
