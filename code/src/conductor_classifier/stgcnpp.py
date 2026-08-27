"""ST-GCN++ (pyskl) — self-contained PyTorch port, no mmcv/pyskl dependency.

Replicates pyskl's STGCN backbone (gcn_adaptive='init', gcn_with_res=True,
tcn_type='mstcn', graph coco/spatial) plus GCNHead, with module/parameter
names identical to pyskl so that an official checkpoint, e.g.
``stgcnpp_ntu60_xsub_hrnet_j.pth``, loads with ``strict=True``.

Sources ported from (kennymckormick/pyskl @ main):
  - pyskl/utils/graph.py            (Graph, spatial mode, normalize_digraph)
  - pyskl/models/gcns/utils/gcn.py  (unit_gcn)
  - pyskl/models/gcns/utils/tcn.py  (unit_tcn, mstcn)
  - pyskl/models/gcns/stgcn.py      (STGCNBlock, STGCN)
  - pyskl/models/heads/simple_head.py (GCNHead)
  - configs/stgcn++/stgcn++_ntu60_xsub_hrnet/j.py (backbone/head args)

Input convention (pyskl FormatGCNInput): keypoint tensor of shape
(N, M, T, V, C) with V=17 COCO joints and C=3 (x, y, confidence).
"""

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-4


# ---------------------------------------------------------------------------
# Graph construction (pyskl/utils/graph.py)
# ---------------------------------------------------------------------------

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


class Graph:
    """Skeleton graph. Only what ST-GCN++/coco/'spatial' needs is ported."""

    def __init__(self, layout='coco', mode='spatial', max_hop=1,
                 num_node=None, inward=None):
        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode

        if layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
        elif layout == 'custom':
            # Extension to use our own joint set as-is (e.g. both 51) — for
            # architecture-comparison experiments. inward = [(child, parent), ...],
            # indices are local to the set.
            assert num_node and inward, 'custom layout needs num_node and inward'
            self.num_node = int(num_node)
            self.inward = [tuple(e) for e in inward]
        else:
            raise AssertionError('only coco / custom layouts are ported')
        self.center = 0
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

        assert mode == 'spatial', 'only the spatial mode is ported'
        self.A = self.spatial()

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A


# ---------------------------------------------------------------------------
# unit_gcn (pyskl/models/gcns/utils/gcn.py)
# ---------------------------------------------------------------------------

class unit_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, A, adaptive='init',
                 conv_pos='pre', with_res=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init'], 'only None/init adaptivity is ported'
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        if self.adaptive == 'init':
            # learnable adjacency, initialized from the static graph
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels))
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        A = self.A  # adaptive in [None, 'init']

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)


# ---------------------------------------------------------------------------
# unit_tcn / mstcn (pyskl/models/gcns/utils/tcn.py)
# ---------------------------------------------------------------------------

class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 dilation=1, norm='BN', dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class mstcn(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride,
                         dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act,
            nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)


# ---------------------------------------------------------------------------
# STGCNBlock / STGCN backbone (pyskl/models/gcns/stgcn.py)
# ---------------------------------------------------------------------------

class STGCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class STGCN(nn.Module):
    """pyskl STGCN backbone (ST-GCN++ variant via kwargs)."""

    def __init__(self,
                 graph_cfg=dict(layout='coco', mode='spatial'),
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [dict(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1,
                                  residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(),
                                      stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x


# ---------------------------------------------------------------------------
# GCNHead (pyskl/models/heads/simple_head.py, mode='GCN', dropout=0)
# ---------------------------------------------------------------------------

class GCNHead(nn.Module):

    def __init__(self, num_classes, in_channels, dropout=0.):
        super().__init__()
        self.dropout_ratio = dropout
        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else None
        self.in_c = in_channels
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def pool(self, x):
        """(N, M, C, T, V) -> (N, C): GAP over T,V then mean over persons M."""
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)
        x = x.mean(dim=(-2, -1))  # AdaptiveAvgPool2d(1) over (T, V)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        if len(x.shape) != 2:
            x = self.pool(x)
        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class STGCNPP(nn.Module):
    """ST-GCN++ (joint stream), stgcn++_ntu60_xsub_hrnet/j.py configuration.

    Backbone: STGCN(gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn',
    graph coco/spatial). Head: GCNHead(num_classes, 256).
    Loads official pyskl checkpoints with ``strict=True``.
    """

    def __init__(self, num_classes=60, in_channels=3, graph_cfg=None):
        super().__init__()
        self.backbone = STGCN(
            graph_cfg=graph_cfg or dict(layout='coco', mode='spatial'),
            in_channels=in_channels,
            gcn_adaptive='init',
            gcn_with_res=True,
            tcn_type='mstcn')
        self.cls_head = GCNHead(num_classes=num_classes, in_channels=256)

    @staticmethod
    def _canonicalize(x):
        """Accept (N, M, T, V, C), (N, T, V, C) or (N, C=3, T, V=17);
        return (N, M, T, V, C)."""
        if x.dim() == 5:
            return x
        if x.dim() == 4:
            if x.shape[-1] == 3:            # (N, T, V, C)
                return x.unsqueeze(1)
            if x.shape[1] == 3:             # (N, C, T, V)
                return x.permute(0, 2, 3, 1).unsqueeze(1)
            raise ValueError(f'ambiguous 4D input shape {tuple(x.shape)}: '
                             'expected (N, T, V, 3) or (N, 3, T, V)')
        raise ValueError(f'expected 4D or 5D input, got {tuple(x.shape)}')

    def forward_features(self, x):
        """Return the pooled (N, 256) embedding before the classifier."""
        x = self._canonicalize(x)
        feat = self.backbone(x)          # (N, M, 256, T', V)
        return self.cls_head.pool(feat)  # (N, 256)

    def forward(self, x):
        """Return classification logits of shape (N, num_classes)."""
        x = self._canonicalize(x)
        feat = self.backbone(x)
        return self.cls_head(feat)

    def load_pyskl_checkpoint(self, path, strict=True):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        sd = ckpt.get('state_dict', ckpt)
        self.load_state_dict(sd, strict=strict)
        return self


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(here, '..', '..', 'pretrained',
                             'stgcnpp_ntu60_xsub_hrnet_j.pth')
    ckpt_path = os.path.abspath(ckpt_path)

    model = STGCNPP(num_classes=60)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = ckpt.get('state_dict', ckpt)
    model.load_state_dict(sd, strict=True)
    print(f'[OK] strict=True load of {len(sd)} keys from {ckpt_path}')

    model.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 1, 50, 17, 3)  # (N, M, T, V, C)
    with torch.no_grad():
        logits = model(x)
        emb = model.forward_features(x)

    print(f'input   : {tuple(x.shape)}  (N, M, T, V, C)')
    print(f'logits  : {tuple(logits.shape)}   finite: {torch.isfinite(logits).all().item()}')
    print(f'features: {tuple(emb.shape)}  finite: {torch.isfinite(emb).all().item()}')
    top5 = logits.topk(5, dim=1).indices
    print(f'top-5 logit indices per sample: {top5.tolist()}')
