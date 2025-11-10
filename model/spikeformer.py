from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.activation_based.neuron import LIFNode

from module import *


class SpikeDrivenTransformer(nn.Module):
    def __init__(
            self,
            pps_size_h=8,
            img_size_h=32,
            img_size_w=128,
            patch_size=2,
            in_channels=2,
            num_classes=11,
            embed_dims=512,
            num_heads=8,
            mlp_ratios=4,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=2,
            sr_ratios=0,
            T=4,
            pooling_stat="1111",
            spike_mode="lif",
            TET=False,
            pretrained=False,
            pretrained_cfg=None,
            dataset='DEAP',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.img_size_h = img_size_h
        self.T = T
        self.TET = TET
        self.dataset = dataset
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule


        patch_embed = SepConvMix(in_channels=in_channels,
                                 embed_dims=embed_dims,
                                 img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size)
        patch_embed_pps = SepConv_pps(
            in_channels=in_channels,
            embed_dims=embed_dims,
            img_size_h=pps_size_h,
            img_size_w=img_size_w, )
        patch_embed_pps_hci = SepConv_pps_hci(
            in_channels=in_channels,
            embed_dims=embed_dims,
            img_size_h=pps_size_h,
            img_size_w=img_size_w, )

        fusion_blocks = nn.ModuleList(
            [
                nn.ModuleList([
                    TCF_Attention_EEG(
                        dim=embed_dims,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios,
                        layer=j,
                    ),
                    TCF_Attention_PPS(
                        dim=embed_dims,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios,
                        layer=j,
                    ),
                    KV_Cross_Attention_E2P(
                        dim=embed_dims,
                        num_heads=num_heads,
                        layer=j,
                    ),
                    KV_Cross_Attention_P2E(
                        dim=embed_dims,
                        num_heads=num_heads,
                        layer=j,
                    )
                ])
                for j in range(depths)
            ]
        )

        mlp_hidden_dim = int(embed_dims * mlp_ratios)

        mlp = MLP(in_features=embed_dims,
                  hidden_features=mlp_hidden_dim,
                  drop_path=drop_path_rate,
                  layer=0, )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"patch_embed_pps", patch_embed_pps)
        setattr(self, f"patch_embed_pps_hci", patch_embed_pps_hci)

        setattr(self, f"fusion_blocks", fusion_blocks)
        setattr(self, f"mlp", mlp)
        # classification head

        self.head_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x_eeg, x_pps, hook=None):

        fusion_blocks = getattr(self, f"fusion_blocks")
        patch_embed = getattr(self, f"patch_embed")
        patch_embed_pps = getattr(self, f"patch_embed_pps")
        patch_embed_pps_hci = getattr(self, f"patch_embed_pps_hci")
        mlp = getattr(self, f"mlp")
        x_eeg, hook = patch_embed(x_eeg, hook=hook)
        if self.dataset == 'DEAP':
            x_pps, hook = patch_embed_pps(x_pps, hook=hook)
        else:
            x_pps, hook = patch_embed_pps_hci(x_pps, hook=hook)



        for attn_eeg, attn_pps, attn_e2p, attn_p2e in fusion_blocks:

            x_eeg, hook, attn_t_eeg, attn_c_eeg = attn_eeg(x_eeg, hook=hook)
            x_pps, hook, attn_t_pps, attn_c_pps = attn_pps(x_pps, hook=hook)

            out_e2p, hook = attn_e2p(x_eeg, attn_t_pps.to(torch.bool), attn_c_pps.to(torch.bool), hook=hook)
            out_p2e, hook = attn_p2e(x_pps, attn_t_eeg.to(torch.bool), attn_c_eeg.to(torch.bool), hook=hook)

            x_eeg = x_eeg + out_e2p
            x_pps = x_pps + out_p2e

        x_eeg = x_eeg.flatten(3)
        x_pps = x_pps.flatten(3)
        x = torch.cat([x_eeg, x_pps], dim=3)
        x, hook = mlp(x, hook=hook)

        x = x.mean(3)

        return x, hook

    def forward(self, x, hook=None):
        x = x.unsqueeze(1)

        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1,
                                        1)
        else:
            x = x.transpose(0, 1).contiguous()

        x_eeg = x[:, :, :, :self.img_size_h, :]
        x_pps = x[:, :, :, self.img_size_h:, :]


        x, hook = self.forward_features(x_eeg, x_pps, hook=hook)
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if not self.TET:
            x = x.mean(0)  # T 维度上平均 [b,10]

        return x, hook


@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()  # 调用 _cfg() 函数生成的默认配置信息赋值给模型对象 model 的 default_cfg 属性。model.default_cfg 有 timm 库中的 Vision Transformer 模型的默认配置信息
    return model
