import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tdnn import xvecTDNN, Fp32GroupNorm

class YVectorEncoder(nn.Module):
    def __init__(self, dropout = 0.0, affine_group_norm = True):
        super(YVectorEncoder, self).__init__()

        def block(n_in, n_out, k, stride, padding = 0):
            '''
            Create block with convolution, dropout, group normalization,  and activation.
            n_in: input channels
            n_out: output channels
            k: kernel size
            stride: stride of convolution   
            activation: activation function
            :group norm is affine
            code of the block is taken from https://github.com/pytorch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec.py
            '''
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False, padding = padding),
                nn.Dropout(p=dropout),
                Fp32GroupNorm(1, n_out, affine=affine_group_norm),
                nn.ReLU()
            )

        self.multiscale_filter_1 = nn.Sequential(block(1, 90, 12, 6),
                                                  block(90, 160, 5, 3))
        self.multiscale_filter_2 = nn.Sequential(block(1, 90, 18, 9),
                                                  block(90, 160, 5, 2, padding=1))
        self.multiscale_filter_3 = nn.Sequential(block(1, 90, 36, 18),
                                                  block(90, 192, 5, 1, padding=2))

        self.downsampling_filter_1 = block(512, 512, 5, 2) # 512 is sum of 160 + 160 + 192 concateneted output channels of multiscale filters
        self.downsampling_filter_2 = block(512, 512, 3, 2) # other hyperparameters are taken from paper
        self.downsampling_filter_3 = block(512, 512, 3, 2, padding = 1)

        self.resid_connection_1 = nn.MaxPool1d(5, 8) # parameters are choosen to match the last output shape
        self.resid_connection_2 = nn.MaxPool1d(3, 4)
        self.resid_connection_3 = nn.MaxPool1d(3, 2, padding = 1)

        class tfse(nn.Module):
            def __init__(self, channels):
                super(tfse, self).__init__()
                self.x_prime_operation = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
                self.y_operation = nn.Sequential(nn.Linear(channels, 1), nn.Sigmoid())
            def forward(self, x):
                x_prime = self.x_prime_operation(x.mean(dim = 2)).unsqueeze(2) * x # input is of shape [Batch, Channel, Time], shape after operation [B, C]
                x_prime_reshaped = x_prime.permute(0, 2, 1).reshape(x_prime.shape[0] * x_prime.shape[2], x_prime.shape[1]) # we want to apply matrix W_2 for each time t channel-wise
                                                                                                                           # so we need to reshape [Batch, Channel, Time] into [B*T, C]
                                                                                                                           # s.t for each pair BatchxTime matrix(vector) W_2 will be applied to corresponding channels
                
                coefs = self.y_operation(x_prime_reshaped)                             # now we have coefficents for each time in each batch in shape of [B*T, 1], 
                coefs = coefs.reshape(x_prime.shape[0], x_prime.shape[2]).unsqueeze(1) # reshape them to [B, 1, T] so multiplication will be correct
                return x_prime * coefs

        self.tfse_1 = tfse(512) 
        self.tfse_2 = tfse(512)
        self.tfse_3 = tfse(512)

    def forward(self, x):
        filtered_1 = self.multiscale_filter_1(x)
        filtered_2 = self.multiscale_filter_2(x)
        filtered_3 = self.multiscale_filter_3(x)
                
        multiscale_filtered = torch.cat((filtered_1, filtered_2, filtered_3), dim = 1)
        resid_1 = self.resid_connection_1(multiscale_filtered)
        
        tfse_1 = self.tfse_1(self.downsampling_filter_1(multiscale_filtered))
        resid_2 = self.resid_connection_2(tfse_1)

        tfse_2 = self.tfse_2(self.downsampling_filter_2(tfse_1))
        resid_3 = self.resid_connection_3(tfse_2)

        tsfe_3 = self.tfse_3(self.downsampling_filter_3(tfse_2))

        #In some cases, len of time dimension doesn't match for these tensors, so we will cut tensors s.t they match in all dimesnions
        min_time_dim = min([tsfe_3.shape[2], resid_1.shape[2], resid_2.shape[2], resid_3.shape[2]])
        tsfe = tsfe[:, :, min_time_dim]
        resid_1 = resid_1[:, :, min_time_dim]
        resid_2 = resid_2[:, :, min_time_dim]
        resid_3 = resid_3[:, :, min_time_dim]

        return torch.cat((tsfe_3, resid_1, resid_2, resid_3), dim = 1)

class YVectorModel(nn.Module):
    def __init__(self):
        super(YVectorModel, self).__init__()
        self.encoder = YVectorEncoder()
        self.tdnn = xvecTDNN(2048, 512)

    def forward(self, x):
        return self.tdnn(self.encoder(x))
