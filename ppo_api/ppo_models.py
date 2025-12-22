# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 复用RayEncoder以保持观测语义一致
from .encoder import RayEncoder


# PPO动作输出数据类
@dataclass
class PPOActOut:
    """
    PPO策略的动作输出数据类
    
    Attributes:
        action: 缩放后的动作，单位为国际标准单位(SI)，形状为[B, A]
        logp: 当前策略下动作的对数概率，形状为[B, 1]
        mu: tanh压缩前的均值，形状为[B, A]
        std: tanh压缩前的标准差，形状为[B, A]
    """
    action: torch.Tensor  # 缩放后的动作，单位为SI，形状[B, A]
    logp: torch.Tensor    # 当前策略下动作的对数概率，形状[B, 1]
    mu: torch.Tensor      # tanh压缩前的均值，形状[B, A]
    std: torch.Tensor     # tanh压缩前的标准差，形状[B, A]


# Tanh对数行列式雅可比函数
def _tanh_log_det_jac(pre_tanh: torch.Tensor) -> torch.Tensor:
    """
    计算tanh函数的对数行列式雅可比矩阵
    
    Args:
        pre_tanh: tanh压缩前的张量，形状为[B, A]
        
    Returns:
        对数行列式雅可比矩阵，形状为[B, A]
    """
    # 稳定计算：2*(log2 - y - softplus(-2y))，按维度求和
    return 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))


# 动作压缩函数
def _squash(mu: torch.Tensor, log_std: torch.Tensor, eps: torch.Tensor, limits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将高斯分布采样的动作通过tanh压缩到指定范围
    
    Args:
        mu: 均值，形状为[B, A]
        log_std: 对数标准差，形状为[B, A]
        eps: 高斯噪声，形状为[B, A]
        limits: 动作限制，形状为[B, A]
        
    Returns:
        (压缩后的动作, 对数概率, 标准差)
    """
    std = log_std.exp()  # 转换为标准差
    pre_tanh = mu + std * eps  # 采样前的动作
    a = torch.tanh(pre_tanh)  # tanh压缩
    log_det = _tanh_log_det_jac(pre_tanh)  # 计算雅可比行列式
    dist = Normal(mu, std)  # 创建高斯分布
    # 计算对数概率：高斯对数概率减去雅可比行列式
    logp = (dist.log_prob(pre_tanh) - log_det).sum(-1, keepdim=True)
    a_scaled = a * limits  # 缩放到指定范围
    return a_scaled, logp, std


# 动作反压缩函数
def _inverse_squash(action_scaled: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    """
    动作的反压缩，将缩放后的动作转换回tanh压缩前的空间
    
    Args:
        action_scaled: 缩放后的动作，形状为[B, A]
        limits: 动作限制，形状为[B, A]
        
    Returns:
        反压缩后的动作，形状为[B, A]
    """
    # 避免atanh溢出，限制动作范围
    a = (action_scaled / limits.clamp_min(1e-12)).clamp(-0.999999, 0.999999)
    # 使用atanh函数进行反压缩
    return 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)


# PPO策略网络
class PPOPolicy(nn.Module):
    """
    共享编码器的高斯策略网络，使用tanh压缩和独立价值头
    
    Args:
        vec_dim: 输入向量维度
        action_dim: 动作维度，默认3
        hidden: 隐藏层大小，默认64
        d_model: 模型维度，默认128
        num_queries: 查询数量，默认4
        num_heads: 注意力头数，默认4
        learnable_queries: 是否使用可学习查询向量，默认True
        log_std_min: 对数标准差最小值，默认-5.0
        log_std_max: 对数标准差最大值，默认2.0
    """

    def __init__(
        self,
        vec_dim: int,
        action_dim: int = 3,
        hidden: int = 64,
        d_model: int = 128,
        *,
        num_queries: int = 4,
        num_heads: int = 4,
        learnable_queries: bool = True,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        # 射线编码器
        self.encoder = RayEncoder(
            vec_dim,
            hidden=hidden,
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            learnable_queries=learnable_queries,
        )
        # 均值网络
        self.mu = nn.Linear(256, action_dim)
        # 对数标准差参数，全局共享，PPO中更稳定
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # 对数标准差的可配置限制
        lo = float(log_std_min)
        hi = float(log_std_max)
        if hi < lo:
            lo, hi = hi, lo
        self._log_std_min = float(lo)
        self._log_std_max = float(hi)

        # 价值函数网络
        self.value = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def _core(self, obs_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        核心网络前向传播
        
        Args:
            obs_vec: 观测向量，形状为[B, vec_dim]
            
        Returns:
            (均值, 对数标准差, 价值)
        """
        g, _, _ = self.encoder(obs_vec)  # 获取全局特征
        mu = self.mu(g)  # 计算均值
        # 扩展对数标准差并限制范围
        log_std = self.log_std.view(1, -1).expand_as(mu).clamp(self._log_std_min, self._log_std_max)
        v = self.value(g)  # 计算价值
        return mu, log_std, v

    @torch.no_grad()
    def act(self, obs_vec: torch.Tensor, limits: torch.Tensor) -> PPOActOut:
        """
        生成动作
        
        Args:
            obs_vec: 观测向量，形状为[B, vec_dim]
            limits: 动作限制，形状为[B, A]
            
        Returns:
            动作输出，包含动作、对数概率、均值和标准差
        """
        mu, log_std, _ = self._core(obs_vec)  # 获取均值和对数标准差
        eps = torch.randn_like(mu)  # 生成高斯噪声
        a_scaled, logp, std = _squash(mu, log_std, eps, limits)  # 压缩动作
        return PPOActOut(action=a_scaled, logp=logp, mu=mu, std=std)

    def evaluate_actions(self, obs_vec: torch.Tensor, actions_scaled: torch.Tensor, limits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作
        
        Args:
            obs_vec: 观测向量，形状为[B, vec_dim]
            actions_scaled: 缩放后的动作，形状为[B, A]
            limits: 动作限制，形状为[B, A]
            
        Returns:
            (对数概率, 熵, 价值)
        """
        mu, log_std, v = self._core(obs_vec)  # 获取均值、对数标准差和价值
        std = log_std.exp()  # 转换为标准差
        # 将动作映射回tanh压缩前的空间
        y = _inverse_squash(actions_scaled, limits)
        dist = Normal(mu, std)  # 创建高斯分布
        log_det = _tanh_log_det_jac(y)  # 计算雅可比行列式
        # 计算对数概率
        logp = (dist.log_prob(y) - log_det).sum(-1, keepdim=True)
        # 从基础高斯分布近似熵
        ent = dist.entropy().sum(-1, keepdim=True)
        return logp, ent, v

