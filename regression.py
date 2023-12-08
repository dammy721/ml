import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# CSVファイルからデータを読み込み
df = pd.read_csv("regression_data.csv")

# 特徴量Xと目的変数yを定義
X = df.iloc[:, 2:]
y = df.iloc[:, 1]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特徴量のスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Lasso": Lasso(),  # L1正則化を使用した線形回帰
    "Ridge": Ridge(),  # L2正則化を使用した線形回帰
    "ElasticNet": ElasticNet(),  # L1とL2正則化の両方を使用した線形回帰
    "Random Forest": RandomForestRegressor(random_state=42),  # 複数の決定木を組み合わせたアンサンブル回帰
    "XGBoost": XGBRegressor(random_state=42),  # 勾配ブースティングに基づく強力なアンサンブル回帰
    "LightGBM": LGBMRegressor(random_state=42),  # XGBoostと似ているが、より高速な勾配ブースティングアンサンブル回帰
    "Linear SVM": SVR(kernel='linear'),  # 線形カーネルを使用したサポートベクターマシン回帰
    "RBF SVM": SVR(kernel='rbf'),  # 放射基底関数(RBF)カーネルを使用したサポートベクターマシン回帰
    "Polynomial SVM": SVR(kernel='poly'),  # 多項式カーネルを使用したサポートベクターマシン回帰
    "Gaussian Process": GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), n_restarts_optimizer=10, random_state=42)  # カーネルに基づくガウス過程回帰
}

# 各モデルのハイパーパラメータグリッド
param_grids = {
    "Lasso": {'alpha': np.logspace(-4, 1, 50)},
    "Ridge": {'alpha': np.logspace(-4, 1, 50)},
    "ElasticNet": {'alpha': np.logspace(-4, 1, 50), 'l1_ratio': [.1, .5, .7, .9, .95, .99, 1]},
    "Random Forest": {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]},
    "XGBoost": {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]},
    "LightGBM": {'n_estimators': [100, 200, 300], 'max_depth': [-1, 10, 20]},
    "Linear SVM": {'C': [0.1, 1, 10]},
    "RBF SVM": {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]},
    "Polynomial SVM": {'C': [0.1, 1, 10], 'degree': [2, 3, 4]},
    "Gaussian Process": {}  # ガウス過程回帰のパラメータチューニングは一般的ではないため空
}

# 各モデルの設定、パラメータチューニング、訓練、評価
tuned_results = []
plt.figure(figsize=(15, 10))

all_predictions = {}

for i, (name, model) in enumerate(models.items(), 1):
    grid_search = GridSearchCV(model, param_grids[name], cv=5)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test_scaled)
    
    # 予測結果を辞書に保存
    all_predictions[name] = predictions

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # チューニング後の結果を保存
    tuned_results.append([name, grid_search.best_params_, mse, r2])

    # 個別の散布図
    plt.subplot(3, 4, i)
    plt.scatter(y_test, predictions)
    plt.title(name)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

# 全体の散布図
plt.subplot(3, 4, 12)
for name, predictions in all_predictions.items():
    plt.scatter(y_test, predictions, label=name)
plt.title("All Models")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()

plt.tight_layout()
plt.show()

# チューニング後の結果をCSVファイルに保存
tuned_results_df = pd.DataFrame(tuned_results, columns=["Model", "Best Parameters", "MSE", "R2"])
tuned_results_df.to_csv("model_performance_with_tuning.csv", index=False)

