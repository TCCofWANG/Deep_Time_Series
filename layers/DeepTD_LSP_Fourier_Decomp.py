import numpy as np
import torch
import torch.nn as nn

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """

    # 随机获取抽取的傅里叶分量的index
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ########## fourier layer #############
class fourier_decomp(nn.Module):
    # FEB
    def __init__(self, in_channels, out_channels,ratio):
        super(fourier_decomp, self).__init__()
        print('fourier_decomp used!')
        """
            基于频率的拆解器
        """


        self.ratio = ratio

        self.scale = (1 / (in_channels * out_channels))

        self.activation = nn.ReLU()


    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)

        f_L = x_ft.shape[-1]
        # Perform Fourier neural operations
        out_ft_s = torch.zeros(B, H, E, f_L, device=x.device, dtype=torch.cfloat)
        out_ft_t = torch.zeros(B, H, E, f_L, device=x.device, dtype=torch.cfloat)

        num = int(f_L*self.ratio)+2  # 切片到f_L*ratio+1

        out_ft_t[:, :, :, :num] = x_ft[:, :, :, :num]
        out_ft_s[:, :, :, num:] = x_ft[:, :, :,num:]

        x_s = torch.fft.irfft(out_ft_s, n=x.size(-1))
        x_t = torch.fft.irfft(out_ft_t, n=x.size(-1))

        x_s = self.activation(x_s)
        x_t = self.activation(x_t)

        return x_s,x_t








