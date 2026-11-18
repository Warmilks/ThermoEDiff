"""Evolution-direction-conditioned residual diffusion model."""
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForMaskedLM

class AlignDiff(nn.Module):
    def __init__(self, bertDir, inSize, tknNum, L, H, A, fcSize=1024, std=1.0):
        super(AlignDiff, self).__init__()
        self.embedder = AutoModelForMaskedLM.from_pretrained(bertDir)
        self.pos_proj = nn.Linear(inSize + 1, H)
        self.backbone = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=H, nhead=A, batch_first=True),
                                              num_layers=L)

        self.fc_linear = nn.Sequential(
            nn.Linear(H, fcSize),
            nn.ReLU(),
            nn.Linear(fcSize, fcSize),
            nn.ReLU(),
            nn.Linear(fcSize, inSize)
        )
        self.lm_head = nn.Sequential(
            nn.Linear(inSize, inSize),
            nn.LayerNorm([inSize]),
            nn.Linear(inSize, tknNum)
        )
        self.tknNum = tknNum
        self.std = std

    def denoise(self, xref, xquery, xt, t):
        # xref: B,1,C, xquery: B,L,C; xt: B,L,C; t: B,1
        qL = xquery.shape[1]

        xseq = torch.cat([xref, xquery, xt], dim=1)  # B,L,C
        xseq = torch.cat([xseq, t[:, None].repeat(1, xseq.shape[1], 1)], dim=-1)  # B,L,C

        xseq = self.pos_proj(xseq)  # B,L,H

        y_logit = self.backbone(xseq)[:, -qL:] + xseq[:, -qL:]  # [...,:-1,:] # + t
        y_logit = self.fc_linear(y_logit)
        return {'y_logit': y_logit, 'tkn_pre': self.lm_head(xquery + xt + y_logit * t[..., None])}

    # DL_ClassifierModel.py 里 class AlignDiff 的 forward 替换/修改为这样

    def forward(self, data, T=0):
        with torch.no_grad():
            # 起点永远需要
            embeddedSeqAArr = self.embedder.esm(**data['tokenizedSeqAArr']).last_hidden_state  # B,L,1280

            # 只有训练才需要终点来构造监督目标 x0
            if T == 0:
                embeddedSeqBArr = self.embedder.esm(**data['tokenizedSeqBArr']).last_hidden_state  # B,L,1280

        xquery = embeddedSeqAArr  # B,L,C

        # ====== 关键：xref 用外部 y，而不是用 B-A 现场算 ======
        xref = data['tokenizedLabArr'].to(xquery.device)  # (B,1280) or (B,1,1280)
        if xref.dim() == 2:
            xref = xref[:, None, :]  # -> (B,1,1280)

        if T == 0:
            x0 = embeddedSeqBArr - embeddedSeqAArr  # supervise target delta (B,L,1280)

            t = torch.tensor(np.random.uniform(size=(x0.shape[0], 1)),
                             dtype=torch.float32, device=x0.device)  # (B,1)
            eps = torch.normal(mean=0, std=self.std, size=x0.shape, device=x0.device)  # (B,L,1280)
            xt = (1 - t[..., None]) * x0 + t[..., None] * eps

            res = self.denoise(xref, xquery, xt, t)
            res['x0'], res['eps'] = x0, eps
            return res

        else:
            # ====== 关键：采样只需要形状，用 xquery.shape 就行 ======
            x_noised = torch.normal(mean=0, std=self.std, size=xquery.shape, device=xquery.device)

            t_samples = torch.linspace(1, 0, T, device=xquery.device)[:, None, None].repeat(1, xquery.shape[0], 1)
            for t in t_samples:
                v = self.denoise(xref, xquery, x_noised, t)['y_logit']
                x_noised = x_noised + v / len(t_samples)

            return {'y_logit': x_noised, 'tkn_pre': self.lm_head(xquery + x_noised)}
