import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class MDMCritic(nn.Module):
    def __init__(self, args, njoints, nfeats, 
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, 
                 activation="gelu"):
        
        super().__init__()
        
        self.mdm = MDM(args, njoints, nfeats,
                 latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers, num_heads=num_heads, 
                 dropout=dropout, activation=activation)
        
        self.mlp = MLP(in_features=196*latent_dim, hidden_features=2*latent_dim, out_features=1)

        

         # Freeze the parameters of the model 训练的时候不要冻结梯度
        # for param in self.parameters():
        #     param.requires_grad = False



    def forward(self, batch_data):
        motion_better, motion_worse = batch_data['motion_better'], batch_data['motion_worse']
        

        encode_better = self.mdm(motion_better) # [batch size * frames(60) * joints(25) * channels(512)]  (b,196,256)
        encode_worse = self.mdm(motion_worse) 
        

        encode_better = encode_better.reshape(encode_better.shape[0], -1) # [batch size * (25*512)]
        encode_worse = encode_worse.reshape(encode_worse.shape[0], -1)
        
        critic_better = self.mlp(encode_better)
        critic_worse = self.mlp(encode_worse)

        critic = torch.cat((critic_better, critic_worse), dim=1)
        
        return critic
    
    def get_score(self, data):
        encode = self.mdm(data) # [batch size * frames(60) * joints(25) * channels(512)]  (b,196,256)
        encode = encode.reshape(encode.shape[0], -1) # [batch size * (25*512)]
        critic = self.mlp(encode)
        return critic
        

class MDM(nn.Module):
    def __init__(self, args, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu",):
        super().__init__()
        self.args = args
        self.njoints = njoints
        self.nfeats = nfeats

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation

        self.input_feats = self.njoints * self.nfeats # 263*1 或者 23*3


        self.input_process = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)


        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)


        
        self.mlp = MLP(in_features=njoints*latent_dim, hidden_features=2*latent_dim, out_features=1)





    def forward(self, x_input):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper 正态噪声
        timesteps: [batch_size] (int)
        """
        x = x_input

        x = self.input_process(x) # 论文图2下面的Linear

        # adding the timestep embed
        xseq = x  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq) 

        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

