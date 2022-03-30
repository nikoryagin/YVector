import torch
import torch.nn as nn
import torch.nn.functional as F

class Fp32GroupNorm(nn.GroupNorm): # taken from https://github.com/pytorch/fairseq/blob/main/fairseq/models/wav2vec
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class TDNN(nn.Module):
    # taken from https://github.com/cvqluu/TDNN/blob/master/tdnn.py
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    dilation=1,
                ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity
        
        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.kernel = nn.Linear(input_dim*context_size, output_dim)


    def forward(self, inputs):
        '''
        input: size (batch, input_features, seq_len)
        outpu: size (batch, new_seq_len, output_features)
        '''
        
        # ----------Convolution = unfold + matmul + fold
        x = inputs
        _, d, _ = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)
        
        # Unfold input into smaller temporal contexts
        x = F.unfold(x, (self.input_dim, self.context_size), 
                     stride=(self.input_dim, 1), 
                     dilation=(1, self.dilation))

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x) # matmul
        
        # transpose to channel first
        x = x.transpose(1, 2)

        return x


class xvecTDNN(nn.Module):
    #partly taken from https://github.com/manojpamk/pytorch_xvectors/blob/master/models.py
    def __init__(self, input_features, output_features, p_dropout = 0, affine = True):
        super(xvecTDNN, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_features, output_dim=512, context_size=5, dilation=1)
        self.norm_layer_1 = Fp32GroupNorm(1, 512, affine=affine)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=5, dilation=2)
        self.norm_layer_2 = Fp32GroupNorm(1, 512, affine=affine)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=7, dilation=3)
        self.norm_layer_3 = Fp32GroupNorm(1, 512, affine=affine)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.norm_layer_4 = Fp32GroupNorm(1, 512, affine=affine)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.norm_layer_5 = Fp32GroupNorm(1, 1500, affine=affine)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000,512)
        self.norm_layer_fc_1 = nn.BatchNorm1d(512, momentum=0.1, affine=affine)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(512, output_features)


    def forward(self, x):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        x = self.dropout_tdnn1(self.norm_layer_1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.norm_layer_2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.norm_layer_3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.norm_layer_4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.norm_layer_5(F.relu(self.tdnn5(x))))

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.norm_layer_fc_1(self.leaky_relu(self.fc1(stats))))
        x = self.fc2(x)

        return x