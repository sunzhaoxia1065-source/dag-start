"""
后处理修正模块（Post-Processing Correction Modules）

完全独立于主模型和 CovariateFusion 的可插拔组件集合，
包含非对称损失函数和残差修正模块。

使用方式:
    # 方案1：仅用非对称损失（训练层面修正偏差方向）
    --model-hyper-params '{"loss": "AsymmetricMSE", "under_weight": 1.5}'

    # 方案2：仅用残差修正模块（输出层面学习残差）
    --model-hyper-params '{"use_residual_correction": true}'

    # 方案3：冻结主模型，只训练残差修正模块（两阶段策略）
    --model-hyper-params '{"use_residual_correction": true, "rc_freeze_main": true}'

    # 方案4：组合使用
    --model-hyper-params '{"loss": "AsymmetricMSE", "use_residual_correction": true}'
"""

import torch
import torch.nn as nn


class AsymmetricMSELoss(nn.Module):
    """非对称MSE损失：低估时惩罚更重，用于修正系统性低估偏差。

    loss = mean(weight * (pred - target)^2)
    当 pred < target（低估）时，weight = under_weight
    当 pred >= target（高估）时，weight = 1.0

    under_weight > 1: 惩罚低估，驱动模型预测更高值
    under_weight = 1: 退化为标准 MSE
    """

    def __init__(self, under_weight=1.5):
        super().__init__()
        self.under_weight = under_weight

    def forward(self, pred, target):
        error = pred - target
        weight = torch.where(error < 0, self.under_weight, 1.0)
        return torch.mean(weight * error ** 2)


class AsymmetricMAELoss(nn.Module):
    """非对称MAE损失：低估时惩罚更重。"""

    def __init__(self, under_weight=1.5):
        super().__init__()
        self.under_weight = under_weight

    def forward(self, pred, target):
        error = pred - target
        weight = torch.where(error < 0, self.under_weight, 1.0)
        return torch.mean(weight * torch.abs(error))


class ResidualCorrection(nn.Module):
    """基于输出幅度的残差修正模块。

    仅根据模型输出值学习一个非线性修正量，不依赖外生变量。
    设计思路：系统性偏差通常是幅度相关的（如高功率区间低估更多），
    因此用 MLP 学习 output → correction 的映射，通过残差连接叠加。

    output_corrected = output + alpha * MLP(output)

    alpha 初始化为较小值(0.1)，保证训练初期不影响主模型输出。
    MLP 的最后一层 bias 初始化为 0，确保初始修正量为 0。

    与 CovariateFusion 的关系：
    - CovariateFusion: 利用未来气象数据做回归补充，输入是 (output, exog_future)
    - ResidualCorrection: 纯粹基于输出幅度学修正，输入只有 output
    - 二者串联无冲突：DAG输出 → CovariateFusion(气象回归) → ResidualCorrection(幅度修正)
      或者 DAG输出 → ResidualCorrection(幅度修正) → 最终输出（不用CovariateFusion时）
    """

    def __init__(self, pred_dim, hidden_dim=32):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.net = nn.Sequential(
            nn.Linear(pred_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_dim),
        )
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, output):
        correction = self.net(output)
        return output + self.alpha * correction


def freeze_module(module):
    """冻结模块的所有参数。"""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """解冻模块的所有参数。"""
    for param in module.parameters():
        param.requires_grad = True
