"""
KGC-UKAN: A Kolmogorov-Arnold U-Net with KAN-Gated Cross-Scale Context
          and Polarity-Aware Uncertainty Skips for Medical Image Segmentation

Module provenance
------------------
  SAKE   (Spline-Adaptive Kolmogorov   : learnable oriented-gradient bank (Sobel = init
          Edge block)                    special case) + spline-gated edge fusion via a
                                          KANLinear gate. Departs from the fixed-Sobel +
                                          fixed-weight EKAN block of UUEKAN: the edge
                                          orientations are learned and the per-channel
                                          fusion weight is a learned 1-D spline of edge
                                          energy, tying the edge branch to the network's
                                          own KAN identity rather than a generic conv add-on.
  KG-CSA (KAN-Gated Cross-Scale ASPP)  : ASPP+gate [AMU-Net], cross-scale ctx [RCGA CSGC],
                                          Mamba-gate -> KAN-gate (new)
  PU-LASk(Polarity-aware Uncertainty   : boundary-distance uncertainty [UUEKAN],
          Linear-Attention Skip)         polarity-aware O(N) attention [PolaFormer]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def to_2tuple(x):
    if isinstance(x, (int, float)):
        return (int(x), int(x))
    return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rt = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rt.floor_()
        return x / keep * rt


class KANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
                .expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2)
                     * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        sol = torch.linalg.lstsq(A, B).solution
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1)
                                     if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base = F.linear(self.base_activation(x), self.base_weight)
        spline = F.linear(self.b_splines(x).view(x.size(0), -1),
                          self.scaled_spline_weight.view(self.out_features, -1))
        return base + spline

    def regularization_loss(self, ra=1.0, re=1.0):
        l1 = self.spline_weight.abs().mean(-1)
        loss_a = l1.sum()
        p = l1 / loss_a
        loss_e = -torch.sum(p * p.log())
        return ra * loss_a + re * loss_e


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        cfg = dict(grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                   scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])
        if not no_kan:
            self.fc1 = KANLinear(in_features, hidden_features, **cfg)
            self.fc2 = KANLinear(hidden_features, out_features, **cfg)
            self.fc3 = KANLinear(hidden_features, out_features, **cfg)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)
        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(out_features)
        self.dwconv_3 = DW_bn_relu(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, -1).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, -1)).reshape(B, N, -1).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, -1)).reshape(B, N, -1).contiguous()
        x = self.dwconv_3(x, H, W)
        return x


# ---- NOVELTY A: SAKE block (Spline-Adaptive Kolmogorov Edge) ----
# Replaces the fixed-Sobel + fixed-weight EKAN block with:
#   (1) a learnable oriented-gradient bank (Sobel x/y is the init special case), and
#   (2) a spline-gated edge fusion: a KANLinear produces a per-channel gate that is
#       itself a learned 1-D spline of edge energy -> edge strength is content-adaptive
#       and KAN-native rather than a constant scalar.
class OrientedGradient(nn.Module):
    """Learnable oriented derivative bank. Initialised to Sobel-x / Sobel-y, so the
    classic Sobel operator is recovered exactly at step 0; orientations then adapt."""
    def __init__(self, dim, n_orient=2):
        super().__init__()
        self.dim = dim
        self.n_orient = n_orient
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        base = torch.stack([sx, sy], 0)                       # 2,3,3
        if n_orient > 2:
            base = torch.cat([base, torch.zeros(n_orient - 2, 3, 3)], 0)
        self.kernels = nn.Parameter(base.clone())             # n_orient,3,3  (learnable)

    def forward(self, xs):
        C = xs.shape[1]
        grads = []
        for o in range(self.n_orient):
            w = self.kernels[o].view(1, 1, 3, 3).repeat(C, 1, 1, 1)
            grads.append(F.conv2d(xs, w, padding=1, groups=C))
        return torch.sqrt(sum(g * g for g in grads) + 1e-6)   # gradient magnitude


class SAKEBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 no_kan=False, use_edge=True, n_orient=2):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.use_edge = use_edge
        if use_edge:
            self.grad = OrientedGradient(dim, n_orient)
            self.edge_proj = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim), nn.GELU())
            self.gate_kan = KANLinear(dim, dim)               # spline gate over edge energy
        self.layer = KANLayer(in_features=dim, hidden_features=dim, drop=drop, no_kan=no_kan)

    def forward(self, x, H, W):
        B, N, C = x.shape
        xn = self.norm1(x)
        if self.use_edge:
            xs = xn.transpose(1, 2).view(B, C, H, W)
            e = self.edge_proj(self.grad(xs))                 # B C H W
            desc = F.adaptive_avg_pool2d(e, 1).flatten(1)     # B C  (edge energy/channel)
            gate = torch.sigmoid(self.gate_kan(desc)).unsqueeze(-1).unsqueeze(-1)
            e = (e * gate).flatten(2).transpose(1, 2)         # spline-gated edge tokens
            xn = xn + e
        kan = self.layer(self.norm2(xn), H, W)
        return x + self.drop_path(kan)


# ---- NOVELTY B: KG-CSA bottleneck ----
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(6, 12, 18)):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for r in rates])
        self.gp = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1),
                                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        feats = [self.b1(x)] + [b(x) for b in self.blocks]
        g = F.interpolate(self.gp(x), size=size, mode='bilinear', align_corners=True)
        feats.append(g)
        return self.fuse(torch.cat(feats, dim=1))


class KGCSA(nn.Module):
    def __init__(self, in_dim, context_dims, rates=(6, 12, 18), proj=128):
        super().__init__()
        self.aspp = ASPP(in_dim, in_dim, rates=rates)
        self.ctx_proj = nn.ModuleList([nn.Conv2d(c, proj, 1) for c in context_dims])
        ctx_vec = proj * len(context_dims)
        self.kan_in = KANLinear(ctx_vec, max(in_dim // 2, proj))
        self.kan_out = KANLinear(max(in_dim // 2, proj), in_dim)

    def forward(self, x, context_feats):
        aspp = self.aspp(x)
        descs = [F.adaptive_avg_pool2d(p(f), 1).flatten(1) for p, f in zip(self.ctx_proj, context_feats)]
        desc = torch.cat(descs, dim=1)
        gate = torch.sigmoid(self.kan_out(self.kan_in(desc))).unsqueeze(-1).unsqueeze(-1)
        return aspp * gate + x


# ---- NOVELTY C: PU-LASk skip ----
class PolarityUncertaintyAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, num_heads=4, kernel_size=5, alpha=4.0):
        super().__init__()
        assert enc_dim % num_heads == 0
        self.enc_dim = enc_dim
        self.num_heads = num_heads
        self.head_dim = enc_dim // num_heads
        self.q = nn.Linear(enc_dim, enc_dim)
        self.kv = nn.Linear(dec_dim, enc_dim * 2)
        self.g = nn.Linear(enc_dim, enc_dim)
        self.proj = nn.Linear(enc_dim, enc_dim)
        self.dwc = nn.Conv2d(self.head_dim, self.head_dim, kernel_size,
                             groups=self.head_dim, padding=kernel_size // 2)
        self.power = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        self.alpha = alpha
        self.scale = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.prob = nn.Conv2d(dec_dim, 1, 1)
        self.tau = 0.5

    def forward(self, enc_skip, dec_feat):
        B, C, H, W = enc_skip.shape
        if dec_feat.shape[2:] != (H, W):
            dec_feat = F.interpolate(dec_feat, size=(H, W), mode='bilinear', align_corners=False)
        N = H * W
        p = torch.sigmoid(self.prob(dec_feat))
        u = (self.tau - (p - self.tau).abs()) / self.tau          # B 1 H W in [0,1]
        u_tok = u.flatten(2).transpose(1, 2)                       # B N 1

        xq = enc_skip.flatten(2).transpose(1, 2)
        xkv = dec_feat.flatten(2).transpose(1, 2)
        q = self.q(xq) * (1.0 + u_tok)
        g = self.g(xq)
        k, v = self.kv(xkv).chunk(2, dim=-1)

        scale = F.softplus(self.scale)
        power = 1 + self.alpha * torch.sigmoid(self.power)
        q = q / scale
        k = k / scale

        def heads(t):
            return t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        q, k, v = heads(q), heads(k), heads(v)

        relu = F.relu
        q_pos, q_neg = relu(q) ** power, relu(-q) ** power
        k_pos, k_neg = relu(k) ** power, relu(-k) ** power
        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k_sim = torch.cat([k_pos, k_neg], dim=-1)
        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k_sim.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_sim.transpose(-2, -1) * (N ** -0.5)) @ (v1 * (N ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k_sim.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_sim.transpose(-2, -1) * (N ** -0.5)) @ (v2 * (N ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1).transpose(1, 2).reshape(B, N, C)
        vv = v.reshape(B * self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2)
        vv = self.dwc(vv).reshape(B, C, N).permute(0, 2, 1)
        x = (x + vv) * g
        x = self.proj(x)
        out = x.transpose(1, 2).reshape(B, C, H, W)
        return out, u


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size); patch_size = to_2tuple(patch_size)
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x): return self.conv(x)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x): return self.conv(x)


class KGC_UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 img_size=224, embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], use_edge=True, use_pulask=True, aspp_rates=(6, 12, 18),
                 **kwargs):
        super().__init__()
        kid = embed_dims[0]
        self.deep_supervision = deep_supervision
        self.use_pulask = use_pulask

        self.encoder1 = ConvLayer(input_channels, kid // 8)
        self.encoder2 = ConvLayer(kid // 8, kid // 4)
        self.encoder3 = ConvLayer(kid // 4, kid)

        self.norm3 = norm_layer(embed_dims[1]); self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1]); self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([SAKEBlock(embed_dims[1], drop_rate, dpr[i], norm_layer,
                                                no_kan, use_edge) for i in range(depths[0])])
        self.block2 = nn.ModuleList([SAKEBlock(embed_dims[2], drop_rate, dpr[sum(depths[:1]) + i],
                                                norm_layer, no_kan, use_edge) for i in range(depths[1])])
        self.dblock1 = nn.ModuleList([SAKEBlock(embed_dims[1], drop_rate, dpr[sum(depths[:2]) + i],
                                                 norm_layer, no_kan, use_edge) for i in range(depths[2])])
        self.dblock2 = nn.ModuleList([SAKEBlock(embed_dims[0], drop_rate, dpr[i], norm_layer,
                                                 no_kan, use_edge) for i in range(depths[0])])

        self.patch_embed3 = PatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])

        self.kgcsa = KGCSA(in_dim=embed_dims[2],
                           context_dims=[kid // 8, kid // 4, kid, embed_dims[1], embed_dims[2]],
                           rates=aspp_rates, proj=128)

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        if use_pulask:
            self.skip4 = PolarityUncertaintyAttention(embed_dims[1], embed_dims[1])
            self.skip3 = PolarityUncertaintyAttention(kid, embed_dims[0])
            self.skip2 = PolarityUncertaintyAttention(kid // 4, embed_dims[0] // 4)
            self.skip1 = PolarityUncertaintyAttention(kid // 8, embed_dims[0] // 8)

        if deep_supervision:
            self.ds4 = nn.Conv2d(embed_dims[1], num_classes, 1)
            self.ds3 = nn.Conv2d(embed_dims[0], num_classes, 1)
            self.ds2 = nn.Conv2d(embed_dims[0] // 4, num_classes, 1)
            self.ds1 = nn.Conv2d(embed_dims[0] // 8, num_classes, 1)

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, 1)

    def forward(self, x):
        B = x.shape[0]; in_size = x.shape[2:]
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); t3 = out

        out, H, W = self.patch_embed3(out)
        for blk in self.block1: out = blk(out, H, W)
        out = self.norm3(out).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous(); t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2: out = blk(out, H, W)
        out = self.norm4(out).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = self.kgcsa(out, context_feats=[t1, t2, t3, t4, out])

        ds = []
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False))
        if self.use_pulask:
            r, _ = self.skip4(t4, out); out = out + r
        else:
            out = out + t4
        if self.deep_supervision:
            ds.append(F.interpolate(self.ds4(out), size=in_size, mode='bilinear', align_corners=False))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1: out = blk(out, H, W)
        out = self.dnorm3(out).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False))
        if self.use_pulask:
            r, _ = self.skip3(t3, out); out = out + r
        else:
            out = out + t3
        if self.deep_supervision:
            ds.append(F.interpolate(self.ds3(out), size=in_size, mode='bilinear', align_corners=False))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2: out = blk(out, H, W)
        out = self.dnorm4(out).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False))
        if self.use_pulask:
            r, _ = self.skip2(t2, out); out = out + r
        else:
            out = out + t2
        if self.deep_supervision:
            ds.append(F.interpolate(self.ds2(out), size=in_size, mode='bilinear', align_corners=False))

        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear', align_corners=False))
        if self.use_pulask:
            r, _ = self.skip1(t1, out); out = out + r
        else:
            out = out + t1
        if self.deep_supervision:
            ds.append(F.interpolate(self.ds1(out), size=in_size, mode='bilinear', align_corners=False))

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear', align_corners=False))
        final = self.final(out)
        if self.deep_supervision and self.training:
            return [final] + ds
        return final

    def regularization_loss(self, ra=1.0, re=1.0):
        return sum(m.regularization_loss(ra, re) for m in self.modules() if isinstance(m, KANLinear))


if __name__ == "__main__":
    for ds in (False, True):
        m = KGC_UKAN(num_classes=1, img_size=256, embed_dims=[128, 160, 256],
                     deep_supervision=ds, use_edge=True, use_pulask=True, aspp_rates=(2, 4, 6))
        m.train()
        x = torch.randn(2, 3, 256, 256)
        y = m(x)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        if isinstance(y, list):
            print(f"[ds={ds}] outs={len(y)} main={tuple(y[0].shape)} params={n:.2f}M")
        else:
            print(f"[ds={ds}] out={tuple(y.shape)} params={n:.2f}M")
        m.eval()
        with torch.no_grad():
            print("        eval:", tuple(m(x).shape), "reg=%.1f" % m.regularization_loss().item())
