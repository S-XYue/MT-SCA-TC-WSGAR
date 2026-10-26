import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import time
import numpy as np

from .backbone import build_backbone, VIT_backbone
from .backbone_inv3 import MyInception_v3, MyRes18
# from .token_encoder import build_token_encoder, TransformerWQ
# from .token_encoder_cross import build_token_encoder, CLIP #TransformerWQ,####token_encoder_cross多标签
# from .token_encoder_cross_ED import build_token_encoder, CLIP   # 2024.6.4 原来的
# from .token_encoder_cross_ED_onetwo import build_token_encoder, CLIP     # 92.67
# from .token_encoder_cross_ED_DRSformer import build_token_encoder, CLIP  # 92.07   92.45  B6
from .token_encoder_cross_ED_DRSformer_pos import build_token_encoder, CLIP  # B2
# from .token_encoder_cross_ED_DAT import build_token_encoder, CLIP


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None,
        out_features=None, act_layer=nn.GELU, drop=0.
    ):
        super().__init__()
        # out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        # x = self.drop(x)
        return x

class DFGAR(nn.Module):
    def __init__(self, args):
        super(DFGAR, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim
        self.num_tokens = args.num_tokens+2

        # feature extraction
        self.backbone = build_backbone(args)
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        self.token_encoder = build_token_encoder(args)
        self.text1 = CLIP(
            256,
            6, 8, 256, 2, 4
        )
        # self.text2 = CLIP(
        #     256,
        #     8, 8, 256, 2, 4
        # )
        # self.text3 = CLIP(
        #     256,
        #     5, 8, 256, 2, 4
        # )
        # act_layer = nn.GELU
        # self.mlp = Mlp(
        #     in_features=8*self.hidden_dim, hidden_features=2*self.hidden_dim,out_features=self.hidden_dim,
        #     act_layer=act_layer, drop=0.1)
        # self.token_encoder1 = TransformerWQ(1024, depth=1, heads=args.nheads_agg, dim_head=128, mlp_dim=1024, dropout=0.1)
        # self.query_embed = nn.Embedding(self.num_tokens, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.token_encoder.d_model, kernel_size=1)
        self.weightagg = FrameWeightFusion(hidden_dim=256)

        if self.dataset == 'volleyball':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        elif self.dataset == 'nba':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
        else:
            assert False

        self.self_attn = nn.MultiheadAttention(self.token_encoder.d_model, args.nheads_agg, dropout=args.drop_rate)
        self.self_attn2 = nn.MultiheadAttention(self.token_encoder.d_model, args.nheads_agg, dropout=args.drop_rate)

        self.dropout1 = nn.Dropout(args.drop_rate)
        self.dropout2 = nn.Dropout(args.drop_rate)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.num_class)
        # self.classifier2 = nn.Linear(2*self.hidden_dim, self.num_class)
        # self.classifier = nn.Linear(6, self.num_class)
        # self.last_linear = nn.Linear(256, 6)
        self.class1 = nn.Linear(256, 2)
        self.class2 = nn.Linear(256, 4)

        self.query_embed1 = nn.Embedding(1, self.hidden_dim)
        self.query_embed2 = nn.Embedding(1, self.hidden_dim)

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.relu = F.relu
        self.gelu = F.gelu

        # self.text_projection = nn.Parameter(torch.empty(256, 256))

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)

        src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        _, c, oh, ow = src.shape

        text = self.text1
        # text = self.text2
        # tt = ['right set', 'right spike', 'right pass', 'right win', 'left set', 'left spike', 'left pass', 'left win']
        tt = read_labels('your/label/path')
        tt = np.hstack(tt).reshape([8, -1])
        tex1 = torch.from_numpy(tt).long()
        # tex = torch.from_numpy(tt).long().cuda()
        tex = text(tex1)

        q1 = self.query_embed1.weight
        q2 = self.query_embed2.weight

        q = torch.cat([q1, q2], dim=0)
        k = v = tex
        q2, _ = self.self_attn2(q, k, v)
        q = q + self.dropout2(q2)
        q = self.norm3(q)

        tex = torch.cat([q, tex], dim=0)

        # tex = tex @ self.text_projection

        src = self.input_proj(src)

        # representations = self.token_encoder(src, None, self.query_embed.weight, pos)
        representations, _ = self.token_encoder(src, None, tex, pos)
        # [1, B x T, K, F'], [1, B x T, K, H' x W']
        # ew = representations[-1]

        representations = representations.reshape(b, t, self.num_tokens, -1)                    # [B, T, K, D]

        if self.dataset == 'volleyball':
            # Aggregation along T dimension (Temporal conv), then K dimension
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)

            representations2 = representations.reshape(b, self.num_tokens, -1, t)
            representations2 = representations2[:, 0:2, :, :]
            representations2, K_maxindex = self.weightagg(representations2)
            representations2 = representations2.permute(1, 0, 2).contiguous()

            representations = torch.mean(representations, dim=2)  # [B x K, D]

            representations = self.norm1(representations)
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)                   # [B, K, D]

            #########self-attn
            representations = representations.permute(1, 0, 2).contiguous()                     # [K, B, D]
            q = representations2
            representations_local = representations[2:, :]
            k = v = representations_local
            representations2, _ = self.self_attn(q, k, v)
            representations = q + self.dropout1(representations2)
            representations = self.norm2(representations)


            representations1 = representations[0, :, :]
            representations2 = representations[1, :, :]

        elif self.dataset == 'nba':
            # Aggregation along T dimension (Temporal conv), then K dimension
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)
            representations = self.conv3(representations)
            representations = self.relu(representations)
            representations = torch.mean(representations, dim=2)                                # [B x K, D]
            representations = self.norm1(representations)
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)                   # [B, K, D]
            representations = representations.permute(1, 0, 2).contiguous()                     # [K, B, D]
            q = k = v = representations
            representations2, _ = self.self_attn(q, k, v)
            representations = representations + self.dropout1(representations2)
            representations = self.norm2(representations)
            representations = representations.permute(1, 0, 2).contiguous()                     # [B, K, D]
            representations = torch.mean(representations, dim=1)                                # [B, D]

        # representations = representations.reshape(b, -1)                                        # [B, K' x F]

        src1 = self.class1(representations1)
        src2 = self.class2(representations2)

        # scr1 = self.last_linear1(representations1)
        # scr = self.last_linear(representations)
        scr = torch.cat([src1, src2], dim=1)
        activities_scores = self.classifier(representations)                                    # [B, C]

        # mask_mat = tex1.contiguous()
        # mask_mat = mask_mat.unsqueeze(0).repeat(b, 1, 1).cuda()
        # scr2 = (scr * mask_mat).sum(-2)
        # scr2 = 3*scr1 + 1*scr2#*0.5
        # activities_scores = activities_scores + activities_scores1

        # activities_scores = torch.mm(representations, tex.t())

        return scr, activities_scores, K_maxindex

class DFGAR_base(nn.Module):
    def __init__(self, args):
        super(DFGAR_base, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim
        self.num_tokens = args.num_tokens

        # feature extraction
        self.backbone = build_backbone(args)
        self.token_encoder = build_token_encoder(args)
        self.query_embed = nn.Embedding(self.num_tokens, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.token_encoder.d_model, kernel_size=1)

        if self.dataset == 'volleyball':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        elif self.dataset == 'nba':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
        else:
            assert False

        self.self_attn = nn.MultiheadAttention(self.token_encoder.d_model, args.nheads_agg, dropout=args.drop_rate)
        self.dropout1 = nn.Dropout(args.drop_rate)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.num_class)

        self.relu = F.relu
        self.gelu = F.gelu

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)

        src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        _, c, oh, ow = src.shape

        src = self.input_proj(src)
        representations, _ = self.token_encoder(src, None, self.query_embed.weight, pos)
        # [1, B x T, K, F'], [1, B x T, K, H' x W']

        representations = representations.reshape(b, t, self.num_tokens, -1)                    # [B, T, K, D]

        if self.dataset == 'volleyball':
            # Aggregation along T dimension (Temporal conv), then K dimension
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)
            representations = torch.mean(representations, dim=2)
            representations = self.norm1(representations)
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)                   # [B, K, D]
            representations = representations.permute(1, 0, 2).contiguous()                     # [K, B, D]
            q = k = v = representations
            representations2, _ = self.self_attn(q, k, v)
            representations = representations + self.dropout1(representations2)
            representations = self.norm2(representations)
            representations = representations.permute(1, 0, 2).contiguous()                     # [B, K, D]
            representations = torch.mean(representations, dim=1)                                # [B, D]
        elif self.dataset == 'nba':
            # Aggregation along T dimension (Temporal conv), then K dimension
            representations = representations.permute(0, 2, 3, 1).contiguous()                  # [B, K, D, T]
            representations = representations.reshape(b * self.num_tokens, -1, t)               # [B x K, D, T]
            representations = self.conv1(representations)
            representations = self.relu(representations)
            representations = self.conv2(representations)
            representations = self.relu(representations)
            representations = self.conv3(representations)
            representations = self.relu(representations)
            representations = torch.mean(representations, dim=2)                                # [B x K, D]
            representations = self.norm1(representations)
            # transformer encoding
            representations = representations.reshape(b, self.num_tokens, -1)                   # [B, K, D]
            representations = representations.permute(1, 0, 2).contiguous()                     # [K, B, D]
            q = k = v = representations
            representations2, _ = self.self_attn(q, k, v)
            representations = representations + self.dropout1(representations2)
            representations = self.norm2(representations)
            representations = representations.permute(1, 0, 2).contiguous()                     # [B, K, D]
            representations = torch.mean(representations, dim=1)                                # [B, D]

        representations = representations.reshape(b, -1)                                        # [B, K' x F]
        activities_scores = self.classifier(representations)                                    # [B, C]

        return activities_scores, activities_scores

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()

        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim

        # feature extraction
        self.backbone = build_backbone(args)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.backbone.num_channels, self.num_class)

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)

        src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        _, c, oh, ow = src.shape

        representations = self.avg_pool(src)
        representations = representations.reshape(b, t, c)
        representations = representations.reshape(b * t, self.backbone.num_channels)        # [B, T, F]
        activities_scores = self.classifier(representations)
        activities_scores = activities_scores.reshape(b, t, -1).mean(dim=1)

        return activities_scores, src

class BaseModel_my(nn.Module):
    def __init__(self, args):
        super(BaseModel_my, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_activities

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim
        self.num_tokens = args.num_tokens
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # feature extraction
        # self.backbone = build_backbone(args)
        # self.backbone = MyInception_v3(transform_input=False,pretrained=True)
        # self.backbone = MyRes18(pretrained=True)
        self.backbone = build_backbone(args)
        self.token_encoder = build_token_encoder(args)

        self.query_embed = nn.Embedding(self.num_tokens, self.hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, self.token_encoder.d_model, kernel_size=1)

        if self.dataset == 'volleyball':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        elif self.dataset == 'nba':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
        elif self.dataset == 'CAD':
            self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
            self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=1)
        else:
            assert False

        self.self_attn = nn.MultiheadAttention(self.token_encoder.d_model, args.nheads_agg, dropout=args.drop_rate)
        self.dropout1 = nn.Dropout(args.drop_rate)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.num_class)

        self.relu = F.relu
        self.gelu = F.gelu

        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)

        ####  reanet18
        src, pos = self.backbone(x)                                                             # [B x T, C, H', W']
        _, c, oh, ow = src.shape
        src = self.input_proj(src)
        representations = src
        # representations, _= self.token_encoder(src, None, src, pos)
        # [1, B x T, K, F'], [1, B x T, K, H' x W']

        ####  inv3
        # outputs = self.backbone(x)
        # representations = []
        # for features in outputs:
        #     if features.shape[2:4] != torch.Size([87, 157]):
        #         features = F.interpolate(features, size=(87, 157), mode='bilinear', align_corners=True)
        #     representations.append(features)
        # src = torch.cat(representations, dim=1)  # B*T, D, OH, OW
        # _, c, oh, ow = src.shape
        # representations = self.input_proj(src)


        representations = representations.permute(0, 1, 3, 2)
        representations = representations.reshape(b, t, 256, oh, ow)

        representations = self.avg_pool(representations)
        representations = representations.reshape(b, t, 256)

        representations = representations.reshape(b * t, 256)        # [B, T, F]
        activities_scores = self.classifier(representations)
        activities_scores = activities_scores.reshape(b, t, -1).mean(dim=1)

        return activities_scores, src

class BaseModel_weak(nn.Module):
    def __init__(self, args):
        super(BaseModel_weak, self).__init__()

        self.dataset = args.dataset
        self.num_class = args.num_activities
        self.input_proj1 = nn.Linear(23*40, 2048)#256*23*40
        self.input_proj2 = nn.Linear(2048, 3493)

        self.backbone = MyRes18(pretrained=True)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, 1, kernel_size=1)

        # model parameters
        self.num_frame = args.num_frame
        self.hidden_dim = args.hidden_dim
        self.num_tokens = args.num_tokens
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, self.num_class)

        # self.backbone = VIT_backbone()



        for name, m in self.named_modules():
            if 'backbone' not in name and 'token_encoder' not in name:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x, mod=False):
        """
        :param x: [B, T, 3, H, W]
        :return:
        """
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)

        src,_ = self.backbone(x)                                                             # [B x T, C, H', W']

        src = self.input_proj(src)
        src = src.reshape(b * t, -1)
        src = self.input_proj1(src)
        if mod:#label
            src_mod = self.input_proj2(src)
            src_mod = src_mod.reshape(b, t, -1).mean(dim=1)
            src_mod = torch.sigmoid(src_mod)


        activities_scores = src.reshape(b, t, -1).mean(dim=1)

        src = self.classifier(src)
        src = src.reshape(b, t, -1).mean(dim=1)


        return activities_scores,src,src_mod


class ADD_GCN_MEM(nn.Module):
    def __init__(self, args):
        super(ADD_GCN_MEM, self).__init__()

        self.input_proj1 = nn.Linear(3493, 2048)  # 256*23*40
        self.input_proj2 = nn.Linear(3493, 8)

        self.backbone = MyRes18(pretrained=True)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, 1, kernel_size=1)
        self.num_classes = 3493
        self.b = args.batch
        self.t = args.num_frame

        self.fc = nn.Conv2d(512, 3493, (1, 1), bias=False)

        self.conv_transform = nn.Conv2d(512, 1024, (1, 1))
        self.relu = nn.LeakyReLU(0.2)

        # self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(1024, 1, 1)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]



    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (BT, C_in, H, W) # C_in: 512
        - Output: (BT, C_out) # C_out: num_classes
        """
        x = self.fc(x)  #(BT, num_classes, H, W)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0]
        x = x.mean(dim=-1)
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape:
        - Input: (BT, C_in, H, W) # C_in: 512
        - Output: (BT, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward_dgcn(self, x):
        # x = self.gcn(x)
        return x

    def forward(self, x, mod=False):
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)
        x, _ = self.backbone(x)




        out1 = self.forward_classification_sm(x)

        v = self.forward_sam(x)  # B*1024*num_classes
        z = v

        out = self.last_linear(z).mean(dim=1)  # B*1*num_classes
        out1 = (out1 + out) / 2
        out1 = out1.reshape(b, t, -1).mean(dim=1)

        activities_scores = self.input_proj1(out1)
        # activities_scores = src.reshape(b, t, -1).mean(dim=1)

        src_mod = torch.sigmoid(out1)
        out1 = self.input_proj2(out1)
        # mask_mat = self.mask_mat.detach()
        # out2 = (out2 * mask_mat).sum(-1)
        # out = (out1 + out) / 2
        # out = out2

        return activities_scores, out1,src_mod

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

class ADD_GCN_label(nn.Module):
    def __init__(self, args):
        super(ADD_GCN_label, self).__init__()

        self.backbone = build_backbone(args)
        self.num_classes = 5#6
        self.b = args.batch
        self.t = args.num_frame

        self.classifier = nn.Linear(self.backbone.num_channels, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.F = nn.Conv2d(768, 512, (1, 1), bias=False)
        self.fc = nn.Conv2d(512, 5, (1, 1), bias=False)#6
        self.fc_emb_1 = nn.Linear(1024*5,512)#6
        self.nl_emb_1 = nn.LayerNorm([512])

        self.fc_activities = nn.Linear(512,9)#8
        # self.token_encoder = TransformerWQ(dim=1024, depth=2, heads=4, dim_head=256, mlp_dim=1024, dropout=0.1)#dim_head=128

        self.conv_transform = nn.Conv2d(512, 1024, (1, 1))
        self.relu = nn.LeakyReLU(0.2)

        # self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(1024, self.num_classes, 1)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # for p in self.backbone.parameters():
        #     p.requires_grad = False

    # def forward_feature(self, x):
    #     b, t, _, h, w = x.shape
    #     x = x.reshape(b * t, 3, h, w)
    #     x, _ = self.backbone(x)
    #     return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (BT, C_in, H, W) # C_in: 512
        - Output: (BT, C_out) # C_out: num_classes
        """
        x = self.fc(x)  #(BT, num_classes, H, W)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0]
        x = x.mean(dim=-1)
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape:
        - Input: (BT, C_in, H, W) # C_in: 512
        - Output: (BT, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward_dgcn(self, x):
        # x = self.gcn(x)
        return x

    def forward(self, x):

        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)
        x, pos = self.backbone(x)

        #inv3
        # x = self.F(x)

        #res连接bb_scores
        # _, c, oh, ow = x.shape
        #
        # bb = self.avg_pool(x)
        # bb = bb.reshape(b, t, c)
        #
        # bb = bb.reshape(b * t, self.backbone.num_channels)  # [B, T, F]
        # bb_scores = self.classifier(bb)
        # bb_scores = bb_scores.reshape(b, t, -1).mean(dim=1)

        # b = self.b
        # t = self.t

        # x = x[:,:,2:18,:]
        out1 = self.forward_classification_sm(x)

        v = self.forward_sam(x)  # B*1024*num_classes

        z = v  # + z
        v = v.permute(0,2,1)

        v = self.token_encoder(v)  # self-attention

        activities_scores = self.fc_emb_1((v.reshape(b*t,-1)))
        activities_scores = self.nl_emb_1(activities_scores)
        activities_scores = self.fc_activities(activities_scores)
        activities_scores = activities_scores.reshape(b, t, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(b, -1)

        activities_scores =activities_scores# + bb_scores

        z = v.permute(0,2,1)###############test

        out2 = self.last_linear(z)  # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
        out = (out1 + out2) / 2
        # out = out2
        out = out.reshape(b, t, -1).mean(dim=1)
        return out, activities_scores

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

class ADD_GCN(nn.Module):
    def __init__(self, args):
        super(ADD_GCN, self).__init__()

        self.backbone = build_backbone(args)
        self.num_classes = 8
        self.b = args.batch
        self.t = args.num_frame

        self.fc = nn.Conv2d(512, 8, (1, 1), bias=False)

        self.conv_transform = nn.Conv2d(512, 1024, (1, 1))
        self.relu = nn.LeakyReLU(0.2)

        # self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(1024, self.num_classes, 1)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    # def forward_feature(self, x):
    #     b, t, _, h, w = x.shape
    #     x = x.reshape(b * t, 3, h, w)
    #     x, _ = self.backbone(x)
    #     return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (BT, C_in, H, W) # C_in: 512
        - Output: (BT, C_out) # C_out: num_classes
        """
        x = self.fc(x)  #(BT, num_classes, H, W)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0]
        x = x.mean(dim=-1)
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape:
        - Input: (BT, C_in, H, W) # C_in: 512
        - Output: (BT, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward_dgcn(self, x):
        # x = self.gcn(x)
        return x

    def forward(self, x):
        b, t, _, h, w = x.shape
        x = x.reshape(b * t, 3, h, w)
        x, _ = self.backbone(x)
        # b = self.b
        # t = self.t

        out1 = self.forward_classification_sm(x)

        v = self.forward_sam(x)  # B*1024*num_classes
        # z = self.forward_dgcn(v)
        z = v  # + z

        out2 = self.last_linear(z)  # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
        out = (out1 + out2) / 2
        # out = out2
        out = out.reshape(b, t, -1).mean(dim=1)
        return out, out

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

def read_labels(path):
    """
    reading annotations for the given sequence
    """
    # annotations1 = [[1, 0, 0, 0, 0, 0],
    #                [0, 1, 0, 0, 0, 0],
    #                [0, 0, 1, 0, 0, 0],
    #                [0, 0, 0, 1, 0, 0],
    #                [0, 0, 0, 0, 1, 0],
    #                [0, 0, 0, 0, 0, 1]
    #                ]
    annotations1 = [[1, 0, 1, 0, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0]]
    annotations2 = [[1, 0, 1, 0, 0, 0],
                   [1, 0, 0, 1, 0, 0],
                   [1, 0, 0, 0, 1, 0],
                   [1, 0, 0, 0, 0, 1],
                   [0, 1, 1, 0, 0, 0],
                   [0, 1, 0, 1, 0, 0],
                   [0, 1, 0, 0, 1, 0],
                   [0, 1, 0, 0, 0, 1]
                   ]
    # annotations2 = [[1, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 1, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 1, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 1, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 1, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 1, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 1, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 1]
    #                ]


    return annotations2


class FrameWeightFusion(nn.Module):
    def __init__(self, hidden_dim, K=2, T=5):    # K=9  78.04    K=3 77.91
        super(FrameWeightFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.K = K  # 表示 K 的维度
        self.T = T  # 表示时间维度 T

        # 权重初始化: 权重矩阵的形状为 [K, T, 1]
        self.frame_weights = nn.Parameter(torch.randn(K, T, 1))  # 随机初始化

        # Softmax 对时间维度进行归一化
        self.softmax = nn.Softmax(dim=1)  # 对 K 维度进行归一化

    def forward(self, representations):  # 输入: [B x K, D, T]
        # print("representations", representations.shape)
        B, K, D, T = representations.size()

        # 计算每个时间步的权重: [K, T, 1] 对应的权重矩阵
        weights = self.softmax(self.frame_weights)  # [K, T, 1] 归一化后的权重   2.更换归一化方式

        # 创建一个新的空 tensor 来存储加权后的表示
        fused_representations = torch.zeros(B, K, D).to(representations.device)

        K_maxindex = []


        # 通过 for 循环遍历 K 对应的每一项
        for k in range(K):
            # 获取当前 K 对应的权重: [T, 1]
            current_weights = weights[k]  # [T, 1]    # 3.更换一下获得位置的方式   SD之后再做一次互注意力

            # max_value_index = torch.argmax(current_weights, dim=0)
            max_value_index = torch.sum(torch.arange(T).to(representations.device) * current_weights.view(-1))  # 78.62
            # 获取对应的 representations: [B, D, T]
            current_representations = representations[:, k, :, :]  # [B, D, T]

            # 对时间维度上的每一帧加权
            weighted_representations = current_representations * current_weights.permute(1, 0)  # [B, D, T]

            # 将加权后的结果按时间维度求和，得到每个加权后的表示
            fused_representations[:, k, :] = torch.sum(weighted_representations, dim=2)  # [B, D]

            K_maxindex.append(max_value_index)
        # print("fused_representations",fused_representations.shape)
        return fused_representations, K_maxindex