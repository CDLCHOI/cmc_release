import clip
from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin
import os

# our version
def build_evaluators(opt):
    movement_enc = MovementConvEncoder(opt['dim_pose']-4, opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])
    text_enc = TextEncoderBiGRUCo(word_size=opt['dim_word'],
                                  pos_size=opt['dim_pos_ohot'],
                                  hidden_size=opt['dim_text_hidden'],
                                  output_size=opt['dim_coemb_hidden'],
                                  device=opt['device'])

    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])

    ckpt_dir = opt['dataset_name']
    # if opt['dataset_name'] == 'humanml':
    #     ckpt_dir = 't2m'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc

def build_evaluators_67(dim_pose, dataset_name, dim_movement_enc_hidden, dim_movement_latent, dim_word, dim_pos_ohot, dim_text_hidden,
                     dim_coemb_hidden, dim_motion_hidden, checkpoints_dir, device):
    movement_enc = MovementConvEncoder(dim_pose, dim_movement_enc_hidden, dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=dim_text_hidden,
                                  output_size=dim_coemb_hidden,
                                  device=device)

    motion_enc = MotionEncoderBiGRUCo(input_size=dim_movement_latent,
                                      hidden_size=dim_motion_hidden,
                                      output_size=dim_coemb_hidden,
                                      device=device)
    contrast_model = MotionCLIP(dim_pose)

    checkpoint = torch.load(os.path.join(checkpoints_dir, dataset_name, 'text_mot_match67', 'model', 'finest.tar'),
                            map_location=device)
    checkpoint_clip = torch.load(os.path.join(checkpoints_dir, dataset_name, 'text_mot_match_clip67', 'model', 'finest.tar'),
                            map_location=device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    contrast_model.load_state_dict(checkpoint_clip['contrast_model'])
    print('Loading Evaluators')
    return text_enc, motion_enc, movement_enc, contrast_model

# our wrapper
class EvaluatorMDMWrapper(object):

    def __init__(self, dataset_name, device):
        opt = {
            'dataset_name': dataset_name,
            'device': device,
            'dim_word': 300,
            'max_motion_length': 196,
            'dim_pos_ohot': len(POS_enumerator),
            'dim_motion_hidden': 1024,
            'max_text_len': 20,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': 263 if dataset_name == 't2m' else 251,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': './checkpoints',
            'unit_length': 4,
        }
        self.dim_pose = opt['dim_pose']
        self.text_encoder, self.motion_encoder, self.movement_encoder = build_evaluators(opt)
        self.opt = opt
        self.device = opt['device']

        self.text_encoder.to(opt['device'])
        self.motion_encoder.to(opt['device'])
        self.movement_encoder.to(opt['device'])

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

        # for param in self.motion_encoder.parameters():  # 可选：彻底冻结参数
        #     param.requires_grad = False
        # for param in self.movement_encoder.parameters():  # 可选：彻底冻结参数
        #     param.requires_grad = False

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        '''
        将一系列变量通过预先定义来做验证的motion_encoder和text_encoder来得到text_embedding和motion_embedding
        '''
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.enable_grad():
            # motions = motions.detach().to(self.device).float()
            motions = motions.to(self.device).float() # 去掉detach

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            # movements = self.movement_encoder(motions[..., :-4]).detach()
            movements = self.movement_encoder(motions[..., :-4]) # 去掉detach
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
    
class Evaluators67(object):

    def __init__(self, dataset_name, device):
        if dataset_name == 't2m':
            dim_pose = 67
        elif dataset_name == 'kit':
            dim_pose = 64
        else:
            raise KeyError('Dataset not Recognized!!!')
        self.dim_pose = dim_pose
        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)
        dim_motion_hidden = 1024
        dim_movement_enc_hidden = 512
        dim_movement_latent = 512
        dim_text_hidden = 512
        dim_coemb_hidden = 512
        checkpoints_dir = 'checkpoints'
        self.unit_length=4

        self.text_encoder, self.motion_encoder, self.movement_encoder, self.contrast_model \
        = build_evaluators_67(dim_pose, dataset_name, dim_movement_enc_hidden, dim_movement_latent, dim_word,
                            dim_pos_ohot, dim_text_hidden, dim_coemb_hidden, dim_motion_hidden, checkpoints_dir, device)
        self.device = device

        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.movement_encoder.to(device)
        self.contrast_model.to(device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
        self.contrast_model.eval()

    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, captions, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            '''clip based'''
            clip_em = self.contrast_model.encode_motion(motions.clone(), m_lens)
            clip_et = self.contrast_model.encode_text(captions)
            clip_em = clip_em / clip_em.norm(dim=1, keepdim=True)
            clip_et = clip_et / clip_et.norm(dim=1, keepdim=True)

            '''original architecture'''
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding
        # return (text_embedding, motion_embedding), (clip_et, clip_em) 这部分是为了算CLIP-score的

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            '''clip based'''
            # clip_em = self.contrast_model.encode_motion(motions.clone(), m_lens)
            # clip_em = clip_em / clip_em.norm(dim=1, keepdim=True)

            '''original architecture'''
            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
        # return motion_embedding, clip_em 这部分是为了算CLIP-score的
    
###########################################################################
###########################################################################
###########################################################################
    
class PositionalEncodingCLIP(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncodingCLIP, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)
    
class MotionEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, ff_size, num_layers, num_heads, dropout, activation):
        super().__init__()
        self.input_feats = in_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncodingCLIP(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)


    def forward(self, motion, padding_mask):
        B, T, D  = motion.shape

        x_emb = self.embed_motion(motion)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=motion.device)][:,None], x_emb], dim=1)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = h.permute(1, 0, 2)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        h = h.permute(1, 0, 2)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        return motion_emb
    
def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def no_grad(nets):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = False
    
class MotionCLIP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.motion_encoder = MotionEncoder(in_dim, 512, 1024, 8, 8, 0.2, 'gelu')
        clip_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        no_grad(self.token_embedding)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            activation="gelu",)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=8)
        self.text_ln = nn.LayerNorm(512)
        self.out = nn.Linear(512, 512)

    def encode_motion(self, motion, m_lens):
        seq_len = motion.shape[1]
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_embedding = self.motion_encoder(motion, padding_mask.to(motion.device))
        return motion_embedding

    def encode_text(self, text):
        device = next(self.parameters()).device

        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.token_embedding(text).float()
            pe_tokens = x + self.positional_embedding.float()
        pe_tokens = pe_tokens.permute(1,0,2)
        out = self.textTransEncoder(pe_tokens)
        out = out.permute(1, 0, 2)
        out = self.text_ln(out)

        out = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        out = self.out(out)
        return out

    def forward(self, motion, m_lens, text):
        motion_features = self.encode_motion(motion, m_lens)
        text_features = self.encode_text(text)

        motion_features = motion_features / motion_features .norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_motion = logit_scale * motion_features @ text_features.t()
        logits_per_text = logits_per_motion.t()
        return logits_per_motion, logits_per_text

    def forward_loss(self, motion, m_lens, text):
        logits_per_motion, logits_per_text = self.forward(motion, m_lens, text)
        labels = torch.arange(len(logits_per_motion)).to(logits_per_motion.device)

        image_loss = F.cross_entropy(logits_per_motion, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        return loss