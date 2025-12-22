# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 辅助函数：1D循环填充
def _circular_pad1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """
    对1D张量进行循环填充
    
    Args:
        x: 输入张量，形状为[B, C, L]或[B, L]
        pad: 填充大小
        
    Returns:
        填充后的张量，形状为[B, C, L+2*pad]或[B, L+2*pad]
    """
    if pad <= 0:
        return x
    left = x[..., -pad:]  # 右侧部分用于左侧填充
    right = x[..., :pad]  # 左侧部分用于右侧填充
    return torch.cat([left, x, right], dim=-1)  # 拼接形成循环填充


# Squeeze-Excitation 1D模块
class SqueezeExcite1D(nn.Module):
    """
    1D Squeeze-Excitation模块，用于通道注意力机制
    
    Args:
        ch: 输入通道数
        r: 压缩比，默认4
    """
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        hid = max(8, ch // r)  # 隐藏层通道数，确保至少为8
        self.fc1 = nn.Linear(ch, hid)  # 压缩：将通道数从ch降至hid
        self.fc2 = nn.Linear(hid, ch)  # 激励：将通道数从hid恢复到ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为[B, C, L]
            
        Returns:
            输出张量，形状为[B, C, L]
        """
        s = x.mean(dim=-1)  # 全局平均池化，形状从[B, C, L]变为[B, C]
        s = F.relu(self.fc1(s))  # 压缩并激活
        s = torch.sigmoid(self.fc2(s))  # 激励，生成通道权重
        return x * s.unsqueeze(-1)  # 应用通道权重，广播到[B, C, L]


# 深度可分离卷积1D模块
class DepthwiseSeparable1D(nn.Module):
    """
    1D深度可分离卷积模块，包含深度卷积和点卷积
    
    Args:
        ch: 通道数
        kernel: 卷积核大小，默认5
        dilation: 膨胀率，默认1
    """
    def __init__(self, ch: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        self.kernel = int(kernel)
        self.dil = int(dilation)
        # 深度卷积：每个通道单独卷积
        self.dw = nn.Conv1d(ch, ch, kernel_size=kernel, groups=ch, bias=False, dilation=self.dil)
        # 点卷积：1x1卷积，用于通道混合
        self.pw = nn.Conv1d(ch, ch, kernel_size=1)
        # 批归一化
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为[B, C, L]
            
        Returns:
            输出张量，形状为[B, C, L]
        """
        # 计算填充大小
        pad = ((self.kernel - 1) * self.dil) // 2
        if pad > 0:
            x = _circular_pad1d(x, pad)  # 循环填充
        out = self.dw(x)  # 深度卷积
        out = F.gelu(out)  # GELU激活
        out = self.pw(out)  # 点卷积
        out = self.bn(out)  # 批归一化
        return out


# 射线处理分支
class RayBranch(nn.Module):
    """
    射线数据处理分支，用于提取LIDAR射线特征
    
    Args:
        in_ch: 输入通道数，默认1
        hidden: 隐藏层通道数，默认64
        layers: 层数，默认4
        kernel: 卷积核大小，默认5
    """
    def __init__(self, in_ch: int = 1, hidden: int = 64, layers: int = 4, kernel: int = 5):
        super().__init__()
        self.in_ch = int(in_ch)
        # 1x1卷积，用于通道扩展
        self.expand = nn.Conv1d(self.in_ch, hidden, kernel_size=1)
        # 膨胀率列表，用于膨胀卷积
        dilations = [1, 2, 4, 8][:layers]
        blocks = []
        # 构建深度可分离卷积块
        for d in dilations:
            blocks += [
                DepthwiseSeparable1D(hidden, kernel=kernel, dilation=d),
                nn.GELU(),
                SqueezeExcite1D(hidden, r=4),
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入射线数据，形状为[B, L]或[B, C, L]
            
        Returns:
            提取的特征，形状为[B, hidden, L]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 从[B, L]扩展为[B, 1, L]
        x = self.expand(x)  # 通道扩展
        x = self.blocks(x)  # 经过深度可分离卷积块
        return x


# 射线编码器
class RayEncoder(nn.Module):
    """
    射线编码器，结合多查询、多头注意力机制
    
    Args:
        vec_dim: 输入向量维度
        hidden: 隐藏层大小，默认64
        d_model: 模型维度，默认128
        num_queries: 查询数量(M)，默认1
        num_heads: 注意力头个数(H)，需满足d_model % H == 0，默认1
        learnable_queries: 是否使用可学习查询向量，默认True
    """

    def __init__(self, vec_dim: int, hidden: int = 64, d_model: int = 128, *, num_queries: int = 1, num_heads: int = 1, learnable_queries: bool = True):
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.learnable_queries = bool(learnable_queries)
        # 姿态特征包含7个维度:
        # [sin_ref, cos_ref, prev_vx/lim, prev_omega/lim, 
        #  dprev_vx/(2*lim), dprev_omega/(2*omega_max), dist_to_task/patch_meters]
        pose_dim = 7
        assert vec_dim >= pose_dim, f"vec_dim必须为N + {pose_dim}，当前为{vec_dim}"
        self.vec_dim = int(vec_dim)
        self.pose_dim = pose_dim
        d_len = vec_dim - self.pose_dim  # 射线数据长度
        
        self.ray_in_ch = 1  # 射线输入通道数
        self.N = max(0, d_len)  # 射线数量
        self.hidden = int(hidden)
        self.d_model = int(d_model)
        assert self.d_model % max(1, self.num_heads) == 0, "d_model必须能被num_heads整除"

        # 射线特征提取分支
        self.br_obs = RayBranch(in_ch=self.ray_in_ch, hidden=hidden)
        # 转换为键和值的卷积层
        self.to_k = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.to_v = nn.Conv1d(hidden, d_model, kernel_size=1)
        # 姿态特征处理MLP
        self.pose_mlp = nn.Sequential(
            nn.Linear(self.pose_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 查询向量处理
        if self.learnable_queries:
            # 可学习查询嵌入，形状为[M, d_model]
            init_scale = 1.0 / math.sqrt(max(1, d_model))
            self.q_params = nn.Parameter(torch.randn(self.num_queries, d_model) * init_scale)
            self.to_q = nn.Identity()  # 恒等变换，避免后续条件判断
        else:
            # 输入条件查询，将姿态嵌入映射到M*d_model
            if self.num_queries > 1:
                self.to_q = nn.Linear(d_model, d_model * self.num_queries)
            else:
                self.to_q = nn.Identity()
        
        # 后续处理MLP
        self.post = nn.Sequential(
            nn.Linear(d_model * 2 + d_model, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )

    def split(self, vec: torch.Tensor):
        """
        将输入向量分割为射线数据和姿态特征
        
        Args:
            vec: 输入向量，形状为[B, vec_dim]
            
        Returns:
            (射线数据, 姿态特征)，分别为[B, N]和[B, pose_dim]
        """
        d_len = self.N * self.ray_in_ch
        d_obs = vec[:, :d_len]  # 射线数据
        pose = vec[:, d_len:d_len + self.pose_dim]  # 姿态特征
        if self.ray_in_ch == 1:
            return d_obs, pose
        x = d_obs.view(d_obs.size(0), self.ray_in_ch, self.N)
        return x, pose

    def forward(self, vec: torch.Tensor):
        """
        前向传播
        
        Args:
            vec: 输入向量，形状为[B, vec_dim]
            
        Returns:
            (全局特征, 键, 值)，全局特征形状为[B, 256]
        """
        # 分割射线数据和姿态特征
        d_obs, pose = self.split(vec)
        # 提取射线特征
        Fmap = self.br_obs(d_obs)  # 形状[B, hidden, N]
        # 转换为键和值
        K = self.to_k(Fmap).transpose(1, 2)  # 形状[B, N, d_model]
        V = self.to_v(Fmap).transpose(1, 2)  # 形状[B, N, d_model]

        # 处理姿态特征
        q_pose = self.pose_mlp(pose)  # 形状[B, d_model]
        
        # 生成查询向量
        if self.learnable_queries:
            # 可学习查询向量：添加姿态嵌入作为批量偏置
            q = self.q_params.unsqueeze(0) + q_pose.unsqueeze(1)  # 形状[B, M, d_model]
        else:
            # 输入条件查询：从姿态嵌入生成
            if self.num_queries > 1:
                qM = self.to_q(q_pose)  # 形状[B, M*d_model]
                q = qM.view(qM.size(0), self.num_queries, self.d_model)  # 形状[B, M, d_model]
            else:
                q = q_pose.view(q_pose.size(0), 1, self.d_model)  # 形状[B, 1, d_model]

        # 多头注意力计算
        H = max(1, self.num_heads)  # 注意力头数
        Dh = self.d_model // H  # 每个头的维度
        # 重塑为多头格式
        K_h = K.view(K.size(0), K.size(1), H, Dh)        # 形状[B, N, H, Dh]
        V_h = V.view(V.size(0), V.size(1), H, Dh)        # 形状[B, N, H, Dh]
        Q_h = q.view(q.size(0), q.size(1), H, Dh)        # 形状[B, M, H, Dh]

        # 注意力机制：对序列长度N进行softmax
        attn_logits = torch.einsum('bmhd,bnhd->bmhn', Q_h, K_h) / math.sqrt(Dh)  # 注意力分数
        attn = torch.softmax(attn_logits, dim=-1)        # 注意力权重，形状[B, M, H, N]
        z_h = torch.einsum('bmhn,bnhd->bmhd', attn, V_h) # 加权求和，形状[B, M, H, Dh]
        z = z_h.reshape(z_h.size(0), z_h.size(1), self.d_model)  # 重塑为[B, M, d_model]

        # 聚合查询结果（取平均）
        z_mean = z.mean(dim=1)        # 形状[B, d_model]
        q_mean = q.mean(dim=1)        # 形状[B, d_model]
        gavg = V.mean(dim=1)          # 形状[B, d_model]

        # 拼接并通过后续MLP
        g = torch.cat([z_mean, gavg, q_mean], dim=-1)  # 形状[B, 3*d_model]
        g = self.post(g)  # 形状[B, 256]
        return g, K, V
