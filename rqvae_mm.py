import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
#import wandb
import random
import collections
from layers import MLPLayers
from rq_mm import ResidualVectorQuantizer
from info_nce import InfoNCE, info_nce
# from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
# from tllib.modules.kernels import GaussianKernel

class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 text_dim=1280,
                 kg_dim=1280,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 cf_embedding = 0,
                 align=0.01,
                 recon=1  
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cf_embedding = cf_embedding
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        self.text_dim = text_dim
        self.kg_dim = kg_dim
        self.align=align
        self.recon=recon
        # self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        # self.encoder = MLPLayers(layers=self.encode_layer_dims,
        #                          dropout=self.dropout_prob,bn=self.bn)

        # text encoder
        self.text_encode_layer_dims = [self.text_dim] + self.layers + [self.e_dim]
        self.text_encoder = MLPLayers(layers=self.text_encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        # kg encoder
        self.kg_encode_layer_dims = [self.kg_dim] + self.layers + [self.e_dim]
        self.kg_encoder = MLPLayers(layers=self.kg_encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        # text rqvae
        self.text_rq = ResidualVectorQuantizer(num_emb_list, e_dim, align=self.align,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        # kg rqvae
        self.kg_rq = ResidualVectorQuantizer(num_emb_list, e_dim, align=self.align,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)    


        self.text_decode_layer_dims = self.text_encode_layer_dims[::-1]
        self.text_decoder = MLPLayers(layers=self.text_decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        self.kg_decode_layer_dims = self.kg_encode_layer_dims[::-1]
        self.kg_decoder = MLPLayers(layers=self.kg_decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn) 
        # self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(kernels=[GaussianKernel(alpha=2 ** -1)])
        self.infonce = InfoNCE()  
    def forward(self, x, y,labels,labels_2, use_sk=True):
        x = self.text_encoder(x)
        y = self.kg_encoder(y)
        x_q,  rq_loss, indices = self.text_rq(x,labels, use_sk=use_sk)
        y_q,  rq_loss_2, indices_2 = self.kg_rq(y,labels_2, use_sk=use_sk)

        text_out = self.text_decoder(x_q)
        kg_out = self.kg_decoder(y_q)
        return text_out, kg_out, rq_loss,rq_loss_2, indices, indices_2, x_q, y_q
    
    def vq_initialization(self,x,y, use_sk=True):
        self.text_rq.vq_ini(self.text_encoder(x))
        self.kg_rq.vq_ini(self.kg_encoder(y))
    @torch.no_grad()
    def get_indices(self, xs,ys,zs, labels,labels_2,labels_3, use_sk=False):
        x_e = self.encoder(xs)
        y_e = self.text_encoder(ys)
        z_e = self.kg_encoder(zs)
        _, _, indices_1 = self.rq(x_e, labels, use_sk=use_sk)
        _, _, indices_2 = self.text_rq(y_e, labels_2, use_sk=use_sk)
        _, _, indices_3 = self.kg_rq(z_e, labels_3, use_sk=use_sk)
        return indices_1,indices_2,indices_3

    def compute_loss(self, text_out,kg_out, quant_loss,quant_loss_2, emb_idx, dense_out,dense_out_2, xs,ys):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(text_out, xs, reduction='mean')+F.mse_loss(kg_out, ys, reduction='mean')

        # elif self.loss_type == 'mmd':
        #     loss_recon = self.mkmmd_loss(out, xs)+self.mkmmd_loss(text_out, ys)+self.mkmmd_loss(kg_out, zs)
        else:
            raise ValueError('incompatible loss type')
        total_loss = self.recon*loss_recon + self.quant_loss_weight * (quant_loss+quant_loss_2)+self.align*(self.infonce(dense_out,dense_out_2))
        # total_loss = self.recon*loss_recon + self.quant_loss_weight * (quant_loss+quant_loss_2)

        return total_loss, None, loss_recon, quant_loss
