import math
import torch
from torch import nn
from torch.nn import functional as F
class SeriesDecomp(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
    def forward(self, x):
        p = self.k // 2
        xp = torch.cat([x[:, :1].expand(-1, p, -1), x, x[:, -1:].expand(-1, p, -1)], 1)
        t = F.avg_pool1d(xp.transpose(1, 2), self.k, 1).transpose(1, 2)
        return x - t, t
class AutoCorrelation(nn.Module):
    def __init__(self, d, h, c):
        super().__init__()
        self.h, self.c = h, c
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
    def forward(self, q, k, v):
        b, n, _ = q.shape
        s = k.shape[1]
        q, k, v = self.q(q).view(b, n, self.h, -1), self.k(k).view(b, s, self.h, -1), self.v(v).view(b, s, self.h, -1)
        k, v = (torch.cat([k, k.new_zeros(b, n - s, *k.shape[2:])], 1), torch.cat([v, v.new_zeros(b, n - s, *v.shape[2:])], 1)) if n > s else (k[:, :n], v[:, :n])
        r = torch.fft.irfft(torch.fft.rfft(q.permute(0, 2, 3, 1), dim=-1) * torch.fft.rfft(k.permute(0, 2, 3, 1), dim=-1).conj(), n=n, dim=-1).mean((1, 2))
        w, tau = r.topk(max(1, int(self.c * math.log(n))), -1)
        vv = v.permute(0, 2, 3, 1).unsqueeze(1).expand(-1, tau.shape[1], -1, -1, -1)
        idx = ((torch.arange(n, device=q.device) - tau[..., None]) % n)[:, :, None, None].expand_as(vv)
        z = (w.softmax(-1)[..., None, None, None] * vv.gather(-1, idx)).sum(1)
        return self.o(z.permute(0, 3, 1, 2).reshape(b, n, -1))
class EncoderLayer(nn.Module):
    def __init__(self, d, h, dFf, k, c, p):
        super().__init__()
        self.att = AutoCorrelation(d, h, c)
        self.ff = nn.Sequential(nn.Conv1d(d, dFf, 1, bias=False), nn.GELU(), nn.Dropout(p), nn.Conv1d(dFf, d, 1, bias=False))
        self.dec1, self.dec2 = SeriesDecomp(k), SeriesDecomp(k)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        x, _ = self.dec1(x + self.drop(self.att(x, x, x)))
        x, _ = self.dec2(x + self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2)))
        return x
class DecoderLayer(nn.Module):
    def __init__(self, d, h, dFf, k, c, p):
        super().__init__()
        self.selfAtt, self.crossAtt = AutoCorrelation(d, h, c), AutoCorrelation(d, h, c)
        self.ff = nn.Sequential(nn.Conv1d(d, dFf, 1, bias=False), nn.GELU(), nn.Dropout(p), nn.Conv1d(dFf, d, 1, bias=False))
        self.dec1, self.dec2, self.dec3 = SeriesDecomp(k), SeriesDecomp(k), SeriesDecomp(k)
        self.trendProj = nn.Conv1d(d, 1, 3, padding=1, padding_mode="circular", bias=False)
        self.drop = nn.Dropout(p)
    def forward(self, x, cross):
        x, t1 = self.dec1(x + self.drop(self.selfAtt(x, x, x)))
        x, t2 = self.dec2(x + self.drop(self.crossAtt(x, cross, cross)))
        x, t3 = self.dec3(x + self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2)))
        return x, self.trendProj((t1 + t2 + t3).transpose(1, 2)).transpose(1, 2)
class Autoformer(nn.Module):
    def __init__(self, seqLen, labelLen, predLen, encMarks, decMarks, d=16, h=2, dFf=32, eLayers=1, dLayers=1, k=25, c=1.0, p=0.05, tau=24.0):
        super().__init__()
        self.labelLen, self.predLen = labelLen, predLen
        self.wMean, self.wLast = nn.Parameter(torch.zeros(predLen)), nn.Parameter(torch.exp(-torch.arange(1, predLen + 1) / tau))
        self.decomp = SeriesDecomp(k)
        self.encTok, self.decTok = nn.Conv1d(1, d, 3, padding=1, padding_mode="circular", bias=False), nn.Conv1d(1, d, 3, padding=1, padding_mode="circular", bias=False)
        self.encMark, self.decMark = nn.Linear(encMarks, d, bias=False), nn.Linear(decMarks, d, bias=False)
        self.encLayers = nn.ModuleList([EncoderLayer(d, h, dFf, k, c, p) for _ in range(eLayers)])
        self.decLayers = nn.ModuleList([DecoderLayer(d, h, dFf, k, c, p) for _ in range(dLayers)])
        self.encNorm, self.decNorm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.proj = nn.Linear(d, 1)
        self.drop = nn.Dropout(p)
    def forward(self, x, mEnc, mDec):
        s, t = self.decomp(x)
        t = torch.cat([t[:, -self.labelLen:], x.mean(1, keepdim=True) * self.wMean[None, :, None] + x[:, -1:] * self.wLast[None, :, None]], 1)
        s = torch.cat([s[:, -self.labelLen:], x.new_zeros(x.shape[0], self.predLen, 1)], 1)
        e = self.drop(self.encTok(x.transpose(1, 2)).transpose(1, 2) + self.encMark(mEnc))
        for blk in self.encLayers:
            e = blk(e)
        e = self.encNorm(e)
        e = e - e.mean(1, keepdim=True)
        y = self.drop(self.decTok(s.transpose(1, 2)).transpose(1, 2) + self.decMark(mDec))
        for blk in self.decLayers:
            y, r = blk(y, e)
            t = t + r
        y = self.decNorm(y)
        y = y - y.mean(1, keepdim=True)
        return (t + self.proj(y))[:, -self.predLen:, 0]
