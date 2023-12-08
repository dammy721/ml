from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.datasets import make_regression

# データセットの生成
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=0)

# ハイパーパラメータ空間の定義
param_space = [
    Real(1e-6, 1e+1, "log-uniform", name='alpha'),
    Real(0, 1, name='l1_ratio')
]

# 目的関数の定義
@use_named_args(param_space)
def objective(alpha, l1_ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
    return -np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))

# ベイズ最適化の実行
result = gp_minimize(objective, param_space, n_calls=50, random_state=0)

# 結果の表示
print("最適なパラメータ: alpha=%.5f, l1_ratio=%.5f" % (result.x[0], result.x[1]))
