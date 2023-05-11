import torch
import torch.nn as nn
from torch import distributions as dist
from . import encoder_latent, decoder
from .occnet import OccupancyNetwork

# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
}

# Decoder dictionary
decoder_dict = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
    'cbatchnorm2': decoder.DecoderCBatchNorm2,
    'batchnorm': decoder.DecoderBatchNorm,
    'cbatchnorm_noresnet': decoder.DecoderCBatchNormNoResnet,
    'cbatchnorm_vn': decoder.VNDecoderCBatchNorm
}
