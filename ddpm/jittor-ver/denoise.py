import jittor as jt
from jittor import nn
from typing import Tuple
import numpy as np
from typing import List, Tuple

class DenoiseDiffusion:
    """
    DDPM的物理概率模型（Jittor实现）
    Params:
        beta: 超参数，定义加噪调度
    """
    def __init__(
            self,
            eps_model: nn.Module,
            T: int,
            beta_1: float = 0.0001,
            beta_T: float = 0.02
        ):
        """
        Args:
            eps_model: UNet去噪模型
            T: 总时间步
        """
        self.eps_model = eps_model
        self.beta = jt.linspace(beta_1, beta_T, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = jt.cumprod(self.alpha).clamp(min_v=1e-10)
        self.T = T
        self.sigma2 = self.beta

    def q_xt_x0(self, x_0: jt.Var, t: jt.Var):
        """
        计算q(x_t|x_0)分布的均值和方差
        """
        alpha_bar = self.alpha_bar[t].reshape((-1, 1, 1, 1))
        alpha_bar = jt.array(alpha_bar)
        mean = (alpha_bar ** 0.5) * x_0
        var = 1.0 - alpha_bar
        return mean, var

    def q_sample(self, x_0: jt.Var, t: jt.Var, eps: jt.Var = None):
        """
        从q(x_t|x_0)采样x_t
        """
        if eps is None:
            eps = jt.randn_like(x_0)
        mean, var = self.q_xt_x0(x_0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, x_t: jt.Var, t: jt.Var):
        """
        使用模型进行去噪采样p_theta(x_{t-1}|x_t)
        """
        eps_theta = self.eps_model(x_t, t)
        alpha_bar = self.alpha_bar.gather(0, t).reshape((-1, 1, 1, 1))
        alpha = self.alpha.gather(0, t).reshape((-1, 1, 1, 1))
        eps_coef = (1 - alpha) / (jt.sqrt(1 - alpha_bar) + 1e-10)
        mean = (x_t - eps_coef * eps_theta) / (jt.sqrt(alpha) + 1e-10)
        var = self.sigma2.gather(0, t).reshape((-1, 1, 1, 1))
        eps = jt.randn(x_t.shape, dtype=x_t.dtype)
        return mean + jt.sqrt(var) * eps

    def loss(self, x_0: jt.Var, noise: jt.Var = None):
        """
        计算预测噪声的MSE损失
        """
        batch_size = x_0.shape[0]
        t = jt.randint(0, self.T, (batch_size,), dtype="int32")
        if noise is None:
            noise = jt.randn_like(x_0)
        x_t = self.q_sample(x_0, t, eps=noise)
        eps_theta = self.eps_model(x_t, t)
        return nn.mse_loss(noise, eps_theta)