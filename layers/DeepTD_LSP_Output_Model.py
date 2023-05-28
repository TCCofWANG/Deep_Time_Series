import numpy as np
import torch
import torch.nn as nn




class Season_Model_block(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, thetas_dim):
        super(Season_Model_block, self).__init__()

        self.thetas_dim = thetas_dim
        self.seq_len = seq_len_q
        self.units = 3*int(self.seq_len)

        self.fc_q_theta = nn.Sequential(
            nn.Linear(seq_len_q, self.units),
            nn.Linear(self.units, self.units),
            nn.Linear(self.units, self.units),
            nn.Linear(self.units, self.units),
            nn.Linear(self.units, thetas_dim, bias=False),
        )

        self.fc_k_theta = nn.Sequential(
            nn.Linear(seq_len_kv, self.units),
            nn.Linear(self.units, self.units),
            nn.Linear(self.units, self.units),
            nn.Linear(self.units, self.units),
            nn.Linear(self.units, thetas_dim, bias=False),
        )

        self.fc_v_theta = nn.Sequential(
            nn.Linear(seq_len_kv, self.units),
            nn.Linear(self.units, self.units),
            nn.Linear(self.units, self.units),
            nn.Linear(self.units, self.units),
            nn.Linear(self.units, thetas_dim, bias=False),
        )

        self.forecast_linspace = self.linear_space(self.seq_len)

        # 添加的新频率参数
        self.F_i = torch.nn.Parameter(torch.rand(self.thetas_dim)).unsqueeze(0)


    def linear_space(self,seq_len):
        horizon = seq_len
        return np.arange(0, horizon) / horizon

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bdl,bld->bll", input, weights)


    def seasonality_model(self, thetas, t, device):
        p = thetas.size()[-1]
        assert p <= thetas.shape[2], 'thetas_dim is too big.'
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)

        t_index = torch.tensor(t).to(device).float().unsqueeze(-1)
        self.F_i = self.F_i.to(device)
        x = 2 * np.pi *torch.matmul(t_index,self.F_i)

        s1_cos = torch.stack([torch.cos(x[:,i]*i) for i in range(p1)])
        s1_sin = torch.stack([torch.sin(x[:,p1+i]*i) for i in range(p2)])
        S = torch.cat([s1_cos, s1_sin])



        seasonality_output = torch.zeros(thetas.shape[0], thetas.shape[1], S.shape[-1]).to(device)
        for i in range(len(thetas)):  # 由于增加了batch维度，这里对batch里面的每个样本都进行一次与矩阵T的相乘
            seasonality_output[i] = thetas[i].mm(S.to(device))
        return seasonality_output

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape

        theta_q = self.fc_q_theta(q.permute(0,2,3,1))
        theta_k = self.fc_k_theta(k.permute(0,2,3,1))
        theta_v = self.fc_v_theta(v.permute(0,2,3,1))


        q_k = (torch.einsum("bhex,bhey->bhxy", theta_q, theta_k))
        q_k = torch.softmax(q_k/self.thetas_dim, dim=-2)
        q_kv = torch.einsum("bhxy,bhey->bhex", q_k,theta_v)



        thetas = q_kv.reshape(B,H*E,self.thetas_dim)
        output = self.seasonality_model(thetas,self.forecast_linspace,q.device)

        return (output, None)


class Season_patch_attention(nn.Module):
    def __init__(self,patch_len,N,d_feature,d_model,d_ff):
        super(Season_patch_attention, self).__init__()
        self.patch_len = patch_len
        self.N = N
        self.d_feature = d_feature
        self.d_model = d_model
        self.d_ff = d_ff

        self.input_linear = nn.Linear(self.d_model,self.d_feature)
        self.season_patch_q = nn.Linear(self.patch_len, self.d_ff)
        self.season_patch_k = nn.Linear(self.patch_len, self.d_ff)
        self.season_patch_v = nn.Linear(self.patch_len, self.d_model)
        self.patch_softmax = nn.Softmax(dim=-1)
        self.patch_back_linear = nn.Linear(self.d_model, self.patch_len)
        self.last_linear = nn.Linear(self.d_feature,self.d_model)

    def forward(self,input):
        # input shape bs*n,patchlen,hidden
        input = self.input_linear(input)  # 压缩hidden，不然塞进bs维度很大
        input = input.view(-1,self.N,self.patch_len,self.d_feature)
        input = input.permute(0,-1,1,2)  # bs,dim,N,patch_len
        input = input.contiguous().view(-1,self.N,self.patch_len)  # bs*dim,N,patchlen

        q = self.season_patch_q(input)
        k = self.season_patch_k(input)
        v = self.season_patch_v(input)
        score = self.patch_softmax(torch.bmm(q,k.permute(0,2,1))/self.d_model**0.5)  # bs*dim,n,n
        attention_patch = torch.bmm(score,v)

        season_patch_back = self.patch_back_linear(attention_patch)
        season_patch = season_patch_back.view(-1,self.d_feature,self.N,self.patch_len).permute(0,-2,-1,1)
        season_patch_output = season_patch.contiguous().view(-1,self.patch_len,self.d_feature)
        season_output = self.last_linear(season_patch_output)

        return season_output


class Trend_patch_attention(nn.Module):
    def __init__(self, patch_len, N, d_model,d_ff):
        super(Trend_patch_attention, self).__init__()
        self.patch_len = patch_len
        self.N = N
        self.d_model = d_model
        self.d_ff = d_ff

        self.trend_patch_q = nn.Linear(self.d_model, self.d_ff)
        self.trend_patch_k = nn.Linear(self.d_model, self.d_ff)
        self.trend_patch_v = nn.Linear(self.d_model, self.d_model)
        self.patch_softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        q = self.trend_patch_q(input)
        k = self.trend_patch_k(input)
        v = self.trend_patch_v(input)
        score = self.patch_softmax(torch.bmm(q, k.permute(0, 2, 1)) / self.d_model)
        attention_patch = torch.bmm(score, v)
        trend_output = attention_patch

        return trend_output

    
    