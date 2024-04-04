import torch.nn as nn
from config import *
import torch.nn.functional as F
from crossmodal.transformer import TransformerEncoder



class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG, self).__init__()
        print(
            "Initializing MAG with beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )
        self.W_hv = nn.Linear(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ha = nn.Linear(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)
        self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.W_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.beta_shift = beta_shift
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        ###################################

        # GRU
        self.v_l = nn.GRU(VISUAL_DIM, TEXT_DIM, num_layers=6, batch_first=True)
        self.a_l = nn.GRU(ACOUSTIC_DIM, TEXT_DIM, num_layers=6, batch_first=True)
        self.l_l = nn.GRU(TEXT_DIM, TEXT_DIM, num_layers=6, batch_first=True)
        self.l_v = nn.GRU(TEXT_DIM, VISUAL_DIM, num_layers=6, batch_first=True)
        self.a_v = nn.GRU(ACOUSTIC_DIM, VISUAL_DIM, num_layers=6, batch_first=True)
        self.l_a = nn.GRU(TEXT_DIM, ACOUSTIC_DIM, num_layers=6, batch_first=True)
        self.v_a = nn.GRU(VISUAL_DIM, ACOUSTIC_DIM, num_layers=6, batch_first=True)


        self.num_heads = args.num_heads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask

        #Crossmodal Attentions
        # self.trans_l_with_a = self.get_network(self_type='la')
        # self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        #####



    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'la', 'lv']:
            embed_dim, attn_dropout = TEXT_DIM, self.attn_dropout
        elif self_type in ['a', 'al', 'av']:
            embed_dim, attn_dropout = ACOUSTIC_DIM, self.attn_dropout_a
        elif self_type in ['v', 'vl', 'va']:
            embed_dim, attn_dropout = VISUAL_DIM, self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

        ###############################
    def forward(self, text_embedding, visual, acoustic):
        ######For MOSI
        #Firstly, use the GRU function for feature processing
        # a_l = self.a_l(acoustic)[0]
        # v_l = self.v_l(visual)[0]
        # #Q, K, V),Two parallel cross attention modules
        # l_with_a = self.trans_l_with_a(text_embedding, a_l, a_l)  # Dimension (L, N, d_l)
        # l_with_v = self.trans_l_with_v(text_embedding, v_l, v_l)  # Dimension (L, N, d_l)
        # #Linear addition of the above two vectors
        # new_text_embedding = (0.5 * l_with_a + 0.5 * l_with_v)
        #
        #####For MOSEI
        # l->v and a->v
        l_v = self.l_v(text_embedding)[0]
        a_v = self.a_v(acoustic)[0]
        v_with_l = self.trans_v_with_l(visual, l_v, l_v)  # Dimension (L, N, d_l)
        v_with_a = self.trans_v_with_a(visual, a_v, a_v)  # Dimension (L, N, d_l)
        new_visual = (0.5*v_with_l + 0.5*v_with_a)
        # #v->a and l->a
        v_a = self.v_a(visual)[0]
        l_a = self.l_a(text_embedding)[0]
        a_with_v = self.trans_a_with_v(acoustic, v_a, v_a)  # Dimension (L, N, d_l)
        a_with_l = self.trans_a_with_l(acoustic, l_a, l_a)  # Dimension (L, N, d_l)
        new_acoustic = (0.5*a_with_v + 0.5*a_with_l)
        #MAG
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((new_visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((new_acoustic, text_embedding), dim=-1)))
        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)
        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)
        acoustic_vis_embedding = alpha * h_m
        embedding_output = self.dropout(self.LayerNorm(acoustic_vis_embedding + text_embedding))

        return embedding_output