import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional

def gather(v: torch.Tensor, t: torch.Tensor):
    """
    返回被索引数据的广播扩展 (B,)->(B, 1, 1, 1)
    Params:
        v: 待索引数据
        t: 索引
    Return:
        v_t(B, 1, 1, 1)
    """
    return torch.gather(v, index=t, dim=0).view(-1, 1, 1, 1)

class DenoiseDiffusion:
    """
    Params:
        beta: 超参数，定义加噪调度。这里使用线性调度，另有余弦调度（适合更高分辨率图像，
            防止后期信息丢失太快）和平方调度（适合快速训练但效果一般）
    """
    def __init__(self,
                 eps_model: nn.Module,
                 T: int,
                 device: torch.device,
                 beta_1: float = 0.0001,
                 beta_T: float = 0.02):
        """
        Params:
            eps_model: epsilon_theta. UNet去噪模型
            T: 总时间步，[0, T-1]
            device: 训练用硬件
        """
        self.eps_model = eps_model
        self.beta = torch.linspace(beta_1, beta_T, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.T = T
        # -------------------------------------------------------------------
        # sigma2是alpha和alpha_bar的表达式，但ddpm论文提到直接使用beta拥有
        # 相似的效果，还有论文是学习方差而非使用固定方差
        # -------------------------------------------------------------------
        self.sigma2 = self.beta
        
    def q_xt_x0(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        即重参数化后得到的公式q(x_t|x_0)，返回的是x_t对应高斯分布的均值和方差
        Params:
            x_0: 初始数据（时间步为0的数据）
            t: 经过t步加噪
        Return:
            mean: x_t的均值
            var: x_t的方差
        """
        alpha_bar = gather(self.alpha_bar, t)
        mean = (alpha_bar ** 0.5) * x_0
        var = 1.0 - alpha_bar
        return mean, var
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        由q(x_t|x_0)得出x_t对应高斯分布后，通过eps采样得到x_t
        Params:
            x_0: 初始数据（时间步为0的数据）
            t: 经过t步加噪
            eps: 形状与x_0相同的采样，默认为标准高斯分布随机采样
        Return:
            x_t: t步加噪数据（时间步为t的数据）
        """
        if eps is None:
            eps = torch.randn_like(x_0)
        mean, var = self.q_xt_x0(x_0, t)
        return mean + (var ** 0.5) * eps
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        使用eps_model预测噪声并去噪，即p_theta(x_{t-1}|x_t)
        Params:
            x_t: 待去噪数据（时间步为t的数据）
            t: 当前数据x_t的时间步
        Return:
            x_{t-1}: 去噪结果
        """
        # 求均值mu，其为eps_theta, x_t, t的关系式
        eps_theta = self.eps_model(x_t, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (x_t - eps_coef * eps_theta)
        # 方差sigma2为固定值sigma2_t
        var = gather(self.sigma2, t)
        eps = torch.randn(x_t.shape, device=x_t.device)
        return mean + (var ** 0.5) * eps
    
    def loss(self, x_0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        损失函数，此处设置为epsilon与epsilon_theta的MSE
        Params:
            x0：来自训练数据的干净的图片
            noise: 加噪过程噪声epsilon~N(0, I)
        Return:
            loss: 真实噪声和预测噪声之间的损失
        """
        batch_size = x_0.shape[0]
        # 随机抽样t，加噪声noise得到x_t，通过eps_model预测噪声为eps_theta
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, eps=noise)
        eps_theta = self.eps_model(x_t, t)
        return F.mse_loss(noise, eps_theta)