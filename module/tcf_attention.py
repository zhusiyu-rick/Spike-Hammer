import torch
import torch.nn as nn
from timm.models.layers import DropPath
from spikingjelly.activation_based.neuron import LIFNode


class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, layer=0, name='None'):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.part = name
        self.dim = dim
        self.num_heads = num_heads
        self.layer = layer
        self.shortcut_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy", step_mode='m')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        # self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')


    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape

        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self.part + self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        if hook is not None:
            hook[self.part + self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)
        if hook is not None:
            hook[self.part + self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()

        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)
        if hook is not None:
            hook[self.part + self._get_name() + str(self.layer) + "_x_after_qk"] = attn.detach()
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        # x = self.proj_lif(x)

        return x, hook, attn

class Channel_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, layer=0, name='None'):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.part = name
        self.dim = dim
        self.num_heads = num_heads
        self.layer = layer
        self.shortcut_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy", step_mode='m')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        # self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')


    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape

        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self.part + self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        if hook is not None:
            hook[self.part + self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)
        if hook is not None:
            hook[self.part + self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()

        q = torch.sum(q, dim=4, keepdim=True)
        attn = self.attn_lif(q)
        if hook is not None:
            hook[self.part + self._get_name() + str(self.layer) + "_x_after_qk"] = attn.detach()
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        # x = self.proj_lif(x)

        return x, hook, attn

class TCF_Attention_EEG(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        layer=0,
    ):
        super().__init__()
        self.attn1 = Token_QK_Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            layer=layer,
            name='EEG'
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity() # 类似dropout的drop path正则化

        self.attn2 = Channel_QK_Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            layer=layer,
            name='EEG'
        )

        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = MLP(
        #     in_features=dim,
        #     hidden_features=mlp_hidden_dim,
        #     drop=drop,
        #     layer=layer,
        # )

    def forward(self, x, hook=None):
        x_res = x
        x_attn, hook, attn_t = self.attn1(x_res, hook=hook)
        x_res = x_attn + x_res
        x_attn, hook, attn_c = self.attn2(x_res, hook=hook)
        x = x_res + x_attn
        x = self.drop_path(x)
        # x, hook = self.mlp(x_attn, hook=hook)
        return x, hook, attn_t, attn_c
class TCF_Attention_PPS(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        layer=0,
    ):
        super().__init__()
        self.attn1 = Token_QK_Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            layer=layer,
            name='PPS'
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity() # 类似dropout的drop path正则化

        self.attn2 = Channel_QK_Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            layer=layer,
            name='PPS'
        )

        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = MLP(
        #     in_features=dim,
        #     hidden_features=mlp_hidden_dim,
        #     drop=drop,
        #     layer=layer,
        # )

    def forward(self, x, hook=None):
        x_res = x
        x_attn, hook, attn_t = self.attn1(x_res, hook=hook)
        x_res = x_attn + x_res
        x_attn, hook, attn_c = self.attn2(x_res, hook=hook)
        x = x_res + x_attn
        x = self.drop_path(x)
        # x, hook = self.mlp(x_attn, hook=hook)
        return x, hook, attn_t, attn_c
class KV_Cross_Attention_E2P(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        norm_layer=nn.LayerNorm,
        layer=0,
    ):
        super().__init__()
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.layer = layer
        self.shortcut_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy", step_mode='m')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)


    def forward(self, x, attn_k, attn_v, hook=None):
        T, B, C, H, W = x.shape

        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_v = attn_k & attn_v

        x = q.mul(k_v)
        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)

        return x, hook

class KV_Cross_Attention_P2E(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        norm_layer=nn.LayerNorm,
        layer=0,
    ):
        super().__init__()
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.layer = layer
        self.shortcut_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy", step_mode='m')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)


    def forward(self, x, attn_k, attn_v, hook=None):
        T, B, C, H, W = x.shape

        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_v = attn_k & attn_v

        x = q.mul(k_v)
        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)

        return x, hook

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop_path=0.0,
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features  # 输出特征维度默认与输入特征维度相同
        hidden_features = hidden_features or in_features  # 隐藏层特征维度默认与输入特征维度相同
        self.res = in_features == hidden_features  # 比较in_features和hidden_features。相同， self.res 设置为 True，表示需要进行残差连接
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)  # 近似的全连接 1*1*channels 的卷积核将数据聚合到1个结果
        self.fc1_bn = nn.BatchNorm1d(hidden_features)

        self.fc1_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)

        self.fc2_lif = LIFNode(tau=2.0, detach_reset=True, backend="cupy", step_mode='m')

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, N = x.shape
        identity = x

        x = self.fc1_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()
        x = self.drop_path(x)
        x = x + identity
        return x, hook