#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Gelu算子实现
公式: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
参考: torch.nn.GELU
"""
import numpy as np
from scipy.special import erf


def impl(input_x):
    """
    Gelu算子实现
    
    参数:
    input_x: 输入张量
    
    返回:
    output: GELU激活后的张量
    """
    # GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = np.sqrt(2.0)
    output = input_x * 0.5 * (1.0 + erf(input_x / sqrt2))
    return output.astype(input_x.dtype)

