import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias






class DeepTD_LSP_Encoder(nn.Module):

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(DeepTD_LSP_Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                #这里是encoder_layer的不断堆叠
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns



class DeepTD_LSP_EncoderLayer(nn.Module):
    """
    DeepTD_LSP_EncoderLayer
    """
    def __init__(self, attention_1,attention_2, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DeepTD_LSP_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        #Fed的对应attention模块，FourierBLock或者MultiWaveletTransform
        self.attention_1 = attention_1
        self.attention_2 = attention_2

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):

        #Frequency Enhanced Block

        # encoder -> encoder_layer -> autocorrelation_layer -> FEB
        x_s,x_t = self.attention_1(
            x, x, x,
            attn_mask=attn_mask
        )

        x_s = self.dropout(x_s)
        y = x_s

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = y + x_s


        x_s = self.attention_2(
            y, y, y,
            attn_mask=attn_mask
        )
        x_s = x_s + y

        return x_s,x_t


class DeepTD_LSP_Decoder(nn.Module):
    """
        DeepTD_LSP_Decoder
    """
    def __init__(self, layers, d_model,d_ff,c_out,norm_layer=None, projection_s=None,projection_t=None):
        super(DeepTD_LSP_Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection_s= projection_s
        self.projection_t= projection_t


    def forward(self, x, cross_s, x_mask=None, cross_mask=None, trend=None,season=None):

        for layer in self.layers:
            x,x_t,x_s = layer(x, cross_s, x_mask=x_mask, cross_mask=cross_mask)
            if trend != None:
                trend = trend + x_t
            if season != None:
                season = season + x_s
        if self.norm is not None:
            x = self.norm(x)

        if self.projection_s is not None:
            #从高维映射回输出维度
            x = self.projection_s(x)

        return x, trend, season

class DeepTD_LSP_DecoderLayer(nn.Module):
    """
    DeepTD_LSP_DecoderLayer
    """
    def __init__(self, Fourier_decomp_1, cross_attention_s,Fourier_decomp_2,
                 season_patch_attention,trend_patch_attention1,
                 trend_patch_attention2,d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DeepTD_LSP_DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.Fourier_decomp_1 = Fourier_decomp_1
        self.cross_attention_s = cross_attention_s
        self.Fourier_decomp_2 = Fourier_decomp_2
        self.season_patch_attention = season_patch_attention
        self.trend_patch_attention1 = trend_patch_attention1
        self.trend_patch_attention2 = trend_patch_attention2

        self.conv1_s = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2_s = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)


        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.projection_season = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross_s,x_mask=None, cross_mask=None):
        x_s,x_t1 = self.Fourier_decomp_1(
            x, x, x,
            attn_mask=x_mask
        )

        x_tt1 = self.trend_patch_attention1(x_t1)
        x_s += x_t1 - x_tt1  # trend_attention 后的残差还给season继续拆


        x_s = self.dropout(x_s)

        # 普通模式
        x_ss = self.dropout(self.cross_attention_s(
            x_s, cross_s, cross_s,
            attn_mask=cross_mask
        ))

        x_sss = self.season_patch_attention(x_ss)

        y_s = x_s - x_sss    # 残差输入接下来的网络进行再学习操作

        y_ss = self.dropout(self.activation(self.conv1_s(y_s.transpose(-1, 1))))
        y_ss = self.dropout(self.conv2_s(y_ss).transpose(-1, 1))
        y_s = y_s + y_ss

        y_pred,x_t2 = self.Fourier_decomp_2(
            y_s, y_s, y_s,
            attn_mask=x_mask
        )

        x_tt2 = self.trend_patch_attention2(x_t2)
        y_pred += x_t2 - x_tt2  # trend_attention 后的残差还给season继续拆

        y_pred = self.dropout(y_pred)


        y_t = x_tt1 + x_tt2

        residual_trend = self.projection(y_t.permute(0, 2, 1)).transpose(1, 2)
        residual_season = self.projection_season(x_sss.permute(0, 2, 1)).transpose(1, 2)

        return y_pred,residual_trend,residual_season



