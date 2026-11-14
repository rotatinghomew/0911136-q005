"""标准库实现的对称正定矩阵与正态抽样工具。"""

import math
import random


def cholesky(matrix):
    """返回下三角 L，使 matrix = L L^T；非正定时抛出 ValueError。"""
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0.0:
                    raise ValueError("矩阵不是正定的")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return lower


def inverse_spd(matrix):
    """用 Cholesky 分解求对称正定矩阵的逆。"""
    size = len(matrix)
    lower = cholesky(matrix)
    inverse = [[0.0] * size for _ in range(size)]
    for column in range(size):
        unit = [0.0] * size
        unit[column] = 1.0
        forward = [0.0] * size
        for i in range(size):
            forward[i] = (unit[i] - sum(lower[i][k] * forward[k] for k in range(i))) / lower[i][i]
        backward = [0.0] * size
        for i in range(size - 1, -1, -1):
            backward[i] = (forward[i] - sum(lower[k][i] * backward[k] for k in range(i + 1, size))) / lower[i][i]
        for i in range(size):
            inverse[i][column] = backward[i]
    return [[0.5 * (inverse[i][j] + inverse[j][i]) for j in range(size)] for i in range(size)]


def normal_draws(covariance, count, seed):
    """从零均值正态分布抽取系数扰动，固定种子保证可复现。"""
    lower = cholesky(covariance)
    size = len(covariance)
    generator = random.Random(seed)
    draws = []
    for _ in range(count):
        standard = [generator.gauss(0.0, 1.0) for _ in range(size)]
        draws.append([sum(lower[i][j] * standard[j] for j in range(size)) for i in range(size)])
    return draws
