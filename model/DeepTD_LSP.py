import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_wo_pos
from layers.DeepTD_LSP_AutoCorrelation import Fourier_Decomp_layer, Season_model_layer
from layers.DeepTD_LSP_Output_Model import Season_Model_block,Season_patch_attention,Trend_patch_attention

from layers.DeepTD_LSP_Fourier_Decomp import fourier_decomp

import numpy as np

from layers.DeepTD_LSP_Autoformer_EncDec import DeepTD_LSP_Decoder, my_Layernorm, DeepTD_LSP_DecoderLayer, DeepTD_LSP_EncoderLayer, DeepTD_LSP_Encoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DeepTD_LSP(nn.Module):
    def __init__(self, configs):
        super(DeepTD_LSP, self).__init__()
        # self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.theta_dims = [3,self.pred_len//2]
        self.decoder_layers_num = configs.d_layers
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.N = int((configs.seq_len-configs.patch_len)/configs.stride)+2
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model
        self.fourier_decomp_ratio = configs.fourier_decomp_ratio

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        # embedding还有的作用就是调整输入维度
        self.enc_embedding = DataEmbedding_wo_pos(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.d_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        encoder_self_att_1 = fourier_decomp(in_channels=configs.d_model,
                                    out_channels=configs.d_model,
                                    ratio=self.fourier_decomp_ratio)

        encoder_self_att_2 = Season_Model_block(in_channels=configs.d_model,
                                                           out_channels=configs.d_model,
                                                           seq_len_q=self.seq_len,
                                                           seq_len_kv=self.seq_len,
                                                           thetas_dim=self.theta_dims[1]
                                                           )

        decoder_self_att_1 = fourier_decomp(in_channels=configs.d_model,
                                    out_channels=configs.d_model,
                                    ratio=self.fourier_decomp_ratio)

        decoder_cross_att_s_total = [
                                    Season_Model_block(in_channels=int(configs.d_model),
                                                             out_channels=int(configs.d_model),
                                                             seq_len_q=self.patch_len,
                                                             seq_len_kv=self.patch_len,
                                                             thetas_dim=int(self.theta_dims[1]*configs.stronger),
                                                             )   for i in range(self.decoder_layers_num)
                                    ]
        decoder_self_att_2 = fourier_decomp(in_channels=configs.d_model,
                                    out_channels=configs.d_model,
                                    ratio=self.fourier_decomp_ratio)



        season_patch_attention = Season_patch_attention(patch_len=self.patch_len,
                                                        N=self.N,
                                                        d_feature=self.d_feature,
                                                        d_model=self.d_model,
                                                        d_ff = configs.d_ff)

        trend_patch_attention1 = Trend_patch_attention(patch_len=self.patch_len,
                                                        N=self.N,
                                                        d_model=self.d_model,
                                                        d_ff = configs.d_ff)

        trend_patch_attention2 = Trend_patch_attention(patch_len=self.patch_len,
                                                        N=self.N,
                                                        d_model=self.d_model,
                                                        d_ff = configs.d_ff)



        self.encoder = DeepTD_LSP_Encoder(
            [
                DeepTD_LSP_EncoderLayer(
                    Fourier_Decomp_layer(
                        encoder_self_att_1,
                        configs.d_model, configs.n_heads),
                    Season_model_layer(
                        encoder_self_att_2,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = \
            DeepTD_LSP_Decoder(
                [
                    DeepTD_LSP_DecoderLayer(
                        Fourier_Decomp_layer(
                            decoder_self_att_1,
                            configs.d_model, configs.n_heads),
                        Season_model_layer(
                            decoder_cross_att_s_total[l],
                            configs.d_model, configs.n_heads),
                        Fourier_Decomp_layer(
                            decoder_self_att_2,
                            configs.d_model, configs.n_heads),
                        season_patch_attention,
                        trend_patch_attention1,
                        trend_patch_attention2,

                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                d_model=configs.d_model, d_ff=configs.d_ff, c_out=configs.c_out,
                norm_layer=my_Layernorm(configs.d_model),
                projection_s=nn.Linear(configs.d_model, configs.c_out, bias=True),
                projection_t=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

        self.trend_linear = nn.Linear(self.N*self.patch_len,self.pred_len)
        self.season_linear = nn.Linear(self.N*self.patch_len,self.pred_len)
        self.padding_patch_layer = nn.ReplicationPad1d((0, configs.stride))

    def forier_decmop(self, x):

        B, S, D = x.shape
        seq = x.permute(0, 2, 1)
        x_ft = torch.fft.rfft(seq, dim=-1)

        f_L = x_ft.shape[-1]

        out_ft_s = torch.zeros(B, D, f_L, device=x.device, dtype=torch.cfloat)
        out_ft_t = torch.zeros(B, D, f_L, device=x.device, dtype=torch.cfloat)

        num = int(f_L*self.fourier_decomp_ratio)+2
        out_ft_t[:, :, :num] = x_ft[:, :, :num]
        out_ft_s[:, :, num:] = x_ft[:, :, num:]

        # Return to time domain

        x_s = torch.fft.irfft(out_ft_s, n=seq.size(-1)).permute(0, 2, 1)
        x_t = torch.fft.irfft(out_ft_t, n=seq.size(-1)).permute(0, 2, 1)

        return x_s, x_t


    def get_patch(self,input):
        # input: bs,seqlen,dim
        dim = input.shape[-1]
        input = input.permute(0, 2, 1)  # bs,dim,seqlen

        input_patch = self.padding_patch_layer(input)  # bs,dim,seqlen + stride
        input_patch = input_patch.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # bs,dim,number,patchlen

        input_patch = input_patch.permute(0, 2, 1, 3).contiguous()  # bs,number,dim,patch_len
        input_patch = input_patch.view(-1, dim, self.patch_len)  # bs*number,dim,patchlen
        input_patch = input_patch.permute(0, 2, 1)  # bs*number,patchlen,dim

        return input_patch

    def re_patch(self,input):

        input = input.contiguous().view(-1,self.N,self.patch_len,self.d_feature)
        B,N,P,D = input.shape
        output = input.view(B,N*P,D)

        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):


        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)


        enc_out_patch = self.get_patch(enc_out)
        x_enc_patch = self.get_patch(x_enc)
        seasonal_init_patch, trend_init_patch = self.forier_decmop(x_enc_patch)
        x_mark_dec_patch = self.get_patch(x_mark_dec[:,:self.seq_len,:])


        dec_out_patch = self.dec_embedding(seasonal_init_patch, x_mark_dec_patch)

        season_init_patch = torch.zeros_like(trend_init_patch)
        residual,trend_part_patch,seasonal_part_patch = self.decoder(dec_out_patch, enc_out_patch, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init_patch,season=season_init_patch)

        trend_output = self.trend_linear(self.re_patch(trend_part_patch).permute(0,2,1)).permute(0,2,1)
        season_output = self.season_linear(self.re_patch(seasonal_part_patch).permute(0,2,1)).permute(0,2,1)


        prediction = trend_output + season_output



        return prediction # [B, L, D]
















