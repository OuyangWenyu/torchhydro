import torch
import torch.nn as nn
import torch.nn.functional as F

# https://arxiv.org/pdf/2410.01201
# https://github.com/axion66/minLSTM-implementation
class MinLSTM(nn.Module):
    """
        Only "parallel mode" is supported for conciseness.
        use log space.

        input shape: [batch, seq_len, in_chn]
        output shape: [batch,seq_len, out_chn]
    """
    def __init__(self, input_size: int, hidden_size: int, device=None, dtype=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size*3,bias=False,device=device, dtype=dtype)


    def forward(self, x_t, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x_t.size(0), self.hidden_size, device=x_t.device, dtype=x_t.dtype)
        seq_len = x_t.shape[1]
        f,i,h = torch.chunk(self.linear(x_t),chunks=3,dim=-1)
        diff = F.softplus(-f) - F.softplus(-i)
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = self.log_g(h_prev)
        log_tilde_h = self.log_g(h)
        log_coeff = log_f.unsqueeze(1)
        log_val = torch.cat([log_h_0.unsqueeze(1), (log_i + log_tilde_h)], dim=1)
        h_t = self.parallel_scan_log(log_coeff,log_val)
        return h_t[:,-seq_len:]

    def parallel_scan_log(self,log_coeffs, log_values):
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0)).squeeze(1)
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1).squeeze(1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h) # will return [batch, seq + 1, chn]

    def g(self,x):
        return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

    def log_g(self,x):
        return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))
