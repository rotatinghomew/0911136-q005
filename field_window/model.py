"""区间删失逻辑模型：worked 只说明区间内至少一天可作业。"""

import math

from .numerics import inverse_spd, normal_draws

SEED = 20251114
DRAW_COUNT = 256


def sigmoid(value):
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _log_one_minus(probability):
    if probability >= 1.0:
        return -40.0
    return math.log1p(-probability)


class Standardizer:
    def __init__(self, rows, indices):
        self.indices = list(indices)
        count = len(rows)
        self.mean = []
        self.scale = []
        for index in self.indices:
            values = [row[index] for row in rows]
            mean = sum(values) / count
            variance = sum((value - mean) ** 2 for value in values) / count
            scale = math.sqrt(variance) or 1.0
            self.mean.append(mean)
            self.scale.append(scale)

    def apply(self, row):
        return [(row[index] - self.mean[position]) / self.scale[position]
                for position, index in enumerate(self.indices)]


class IntervalLogistic:
    """以“区间内至少一天可作业”为正例、单日不可作业为负例的逻辑模型。

    输出层为日级可作业概率，正例似然是 1 - prod(1 - p_day)，因此不会把
    作业区间内每一天都当成已作业，也不会把区间填报日当成标签。
    """

    def __init__(self, l2=1.0, max_iterations=300):
        self.l2 = l2
        self.max_iterations = max_iterations
        self.weights = [0.0]
        self.loss = 0.0
        self.iterations = 0
        self.dimension = 0
        self.covariance = [[1.0]]
        self.draws = [[0.0]] * DRAW_COUNT
        self.standardizer = None

    def fit(self, examples, indices):
        rows = [row for example in examples for row in example["features"]]
        self.standardizer = Standardizer(rows, indices)
        encoded = [(example["label"], [self.standardizer.apply(row) for row in example["features"]])
                   for example in examples]
        dimension = len(indices)
        weights = [0.0] * (dimension + 1)

        def objective(candidate):
            total = 0.0
            gradient = [0.0] * (dimension + 1)
            for label, bags in encoded:
                scores = [candidate[0] + sum(candidate[j + 1] * row[j] for j in range(dimension))
                          for row in bags]
                probabilities = [sigmoid(score) for score in scores]
                if label == 1:
                    log_none = sum(_log_one_minus(probability) for probability in probabilities)
                    none = math.exp(log_none)
                    chance = 1.0 - none
                    total -= math.log(max(chance, 1e-12))
                    coefficient = -(none / max(chance, 1e-12))
                else:
                    total -= sum(_log_one_minus(probability) for probability in probabilities)
                    coefficient = 1.0
                for row, probability in zip(bags, probabilities):
                    factor = coefficient * probability
                    gradient[0] += factor
                    for j in range(dimension):
                        gradient[j + 1] += factor * row[j]
            for j in range(1, dimension + 1):
                total += 0.5 * self.l2 * candidate[j] * candidate[j]
                gradient[j] += self.l2 * candidate[j]
            return total, gradient

        loss, gradient = objective(weights)
        step = 0.5
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            improved = False
            for _ in range(24):
                proposal = [weights[k] - step * gradient[k] for k in range(dimension + 1)]
                next_loss, next_gradient = objective(proposal)
                if next_loss < loss - 1e-12:
                    weights, loss, gradient = proposal, next_loss, next_gradient
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
            step = min(step * 1.08, 2.0)
            if max(abs(value) for value in gradient) < 1e-6:
                break
        self.weights = weights
        self.loss = loss
        self.iterations = iterations
        self.dimension = dimension
        try:
            self.covariance = self._covariance(encoded)
        except ValueError:
            # 后验曲率退化时退回对角近似，仍给出可复现的不确定性。
            scale = 1.0 / max(len(encoded), 1)
            self.covariance = [[scale if i == j else 0.0 for j in range(dimension + 1)]
                               for i in range(dimension + 1)]
        self.draws = normal_draws(self.covariance, DRAW_COUNT, SEED)
        return self

    def _covariance(self, encoded):
        """用高斯—牛顿曲率近似后验协方差；矩阵半正定，退化时由调用方兜底。"""
        dimension = self.dimension
        size = dimension + 1
        hessian = [[0.0] * size for _ in range(size)]
        for label, bags in encoded:
            scores = [self.weights[0] + sum(self.weights[j + 1] * row[j] for j in range(dimension))
                      for row in bags]
            probabilities = [sigmoid(score) for score in scores]
            if label == 1:
                log_none = sum(_log_one_minus(probability) for probability in probabilities)
                none = math.exp(log_none)
                chance = max(1.0 - none, 1e-12)
                weight = none / (chance * chance)
                vector = [0.0] * size
                for row, probability in zip(bags, probabilities):
                    vector[0] += probability
                    for j in range(dimension):
                        vector[j + 1] += probability * row[j]
                for i in range(size):
                    for j in range(size):
                        hessian[i][j] += weight * vector[i] * vector[j]
            else:
                for row, probability in zip(bags, probabilities):
                    variance = probability * (1.0 - probability)
                    point = [probability] + [probability * row[j] for j in range(dimension)]
                    for i in range(size):
                        for j in range(size):
                            hessian[i][j] += variance * point[i] * point[j]
        for j in range(1, size):
            hessian[j][j] += self.l2
        return inverse_spd(hessian)

    def score(self, row):
        standardized = self.standardizer.apply(row)
        return self.weights[0] + sum(self.weights[j + 1] * standardized[j]
                                     for j in range(self.dimension))

    def probability(self, row):
        return sigmoid(self.score(row))

    def draw_probabilities(self, row):
        standardized = self.standardizer.apply(row)
        central = self.weights[0] + sum(self.weights[j + 1] * standardized[j]
                                        for j in range(self.dimension))
        return [sigmoid(central + draw[0] + sum(draw[j + 1] * standardized[j]
                                                for j in range(self.dimension)))
                for draw in self.draws]

    def interval(self, row):
        values = sorted(self.draw_probabilities(row))
        return {"mean": sum(values) / len(values),
                "lower": values[int(0.1 * (len(values) - 1))],
                "upper": values[int(0.9 * (len(values) - 1))]}

    def bag_probability(self, rows):
        if not rows:
            return 0.0
        return 1.0 - math.prod(1.0 - self.probability(row) for row in rows)

    def bag_interval(self, rows):
        if not rows:
            return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
        draws = [self.draw_probabilities(row) for row in rows]
        values = sorted(1.0 - math.prod(1.0 - draws[position][draw]
                                        for position in range(len(rows)))
                        for draw in range(len(self.draws)))
        return {"mean": sum(values) / len(values),
                "lower": values[int(0.1 * (len(values) - 1))],
                "upper": values[int(0.9 * (len(values) - 1))]}
