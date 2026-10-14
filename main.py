"""
利用 CRISP-DM 架構 - 多元線性回歸 (Multiple Linear Regression) 互動網頁
- 下拉選單切換兩種問題: 50 Startups / Boston Housing
- 可任意輸入 Dataset URL (或上傳 CSV), 自選目標變數與特徵
- 公式解 (Normal Equation / OLS) 與 梯度下降法 (Gradient Descent) 比較

執行: streamlit run main.py
"""
from __future__ import annotations

import io
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel, SelectKBest, SequentialFeatureSelector,
                                       f_regression, mutual_info_regression)
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# stlite (瀏覽器版) 內建的 pyarrow 是精簡版, 缺少 sklearn 檢查資料型態時用到的類別, 在此補上空殼
if sys.platform == "emscripten":
    import pyarrow as _pa
    for _n in ("Table", "RecordBatch", "Array", "ChunkedArray"):
        if not hasattr(_pa, _n):
            setattr(_pa, _n, type(_n, (), {}))

# ------------------------------------------------------------------
# 常數: 兩種內建問題
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
PROBLEMS = {
    "🚀 50 Startups - 預測新創公司利潤 (Profit)": {
        "key": "startups",
        "urls": [
            "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv",
        ],
        "local": BASE_DIR / "data" / "50_Startups.csv",
        "target": "Profit",
        "unit": "美元",
    },
    "🏠 Boston Housing - 預測房價中位數 (MEDV)": {
        "key": "boston",
        "urls": [
            "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        ],
        "local": BASE_DIR / "data" / "BostonHousing.csv",
        "target": "medv",
        "unit": "千美元",
    },
}

st.set_page_config(page_title="CRISP-DM 多元線性回歸", page_icon="📈", layout="wide")
head_box = st.container()   # 標題 (稍後填入)
ctrl_box = st.container()   # 主畫面的「特徵篩選與最佳模型」控制面板 (稍後填入)


# ------------------------------------------------------------------
# 資料讀取
# ------------------------------------------------------------------
def _src(url):
    """瀏覽器 (stlite/Pyodide) 中 pandas 無法直接連網, 改用 pyodide.http 取回文字。"""
    if sys.platform == "emscripten" and isinstance(url, str) and url.startswith("http"):
        from pyodide.http import open_url
        return io.StringIO(open_url(url).read())
    return url


@st.cache_data(show_spinner="讀取資料中...")
def read_csv_url(url: str, header_none: bool = False) -> pd.DataFrame:
    kw = dict(header=None) if header_none else {}
    try:
        df = pd.read_csv(_src(url), sep=None, engine="python", **kw)
    except Exception:
        df = pd.read_csv(_src(url), sep=r"\s+", engine="python", **kw)
    if header_none:
        df.columns = [f"x{i}" for i in range(df.shape[1] - 1)] + ["y"]
    return df


@st.cache_data(show_spinner="讀取資料中...")
def load_builtin(key: str) -> tuple[pd.DataFrame, str]:
    cfg = next(c for c in PROBLEMS.values() if c["key"] == key)
    for url in cfg["urls"]:
        try:
            return pd.read_csv(_src(url)), url
        except Exception:
            continue
    return pd.read_csv(cfg["local"]), f"本機備援檔 {cfg['local'].name}"


# ------------------------------------------------------------------
# 資料準備 / 模型
# ------------------------------------------------------------------
def prepare_xy(df: pd.DataFrame, target: str, feats: list[str], drop_first: bool = True, baselines=None):
    """缺失值處理 + 類別變數 One-Hot (drop_first=False: 每個類別一欄 100/010/001; True: 移除一個基準類別以避免共線) -> 回傳 X(DataFrame), y(Series), 類別欄位, 清理後 df"""
    d = df[feats + [target]].copy()
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[target])
    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(d[c])]
    for c in feats:
        if c not in cat_cols:
            d[c] = d[c].fillna(d[c].median())
        else:
            d[c] = d[c].fillna(d[c].mode().iloc[0])
    X = pd.get_dummies(d[feats], columns=cat_cols, drop_first=False, dtype=float)
    if drop_first:  # 每個類別變數移除一個基準類別 (預設第一個, 可由 baselines 指定)
        drop = []
        for c in cat_cols:
            levels = sorted(d[c].astype(str).unique())
            b = (baselines or {}).get(c, levels[0])
            drop.append(f"{c}_{b if b in levels else levels[0]}")
        X = X.drop(columns=[x for x in drop if x in X.columns])
    return X.astype(float), d[target].astype(float), cat_cols, d


def prepare_ar(df: pd.DataFrame, target: str, p: int):
    """自迴歸 AR(p): 以 y(t-1) ... y(t-p) 作為特徵預測 y(t) (依資料列順序視為時間序列)"""
    s = pd.to_numeric(df[target], errors="coerce").dropna().reset_index(drop=True).astype(float)
    lags = pd.concat({f"y(t-{k})": s.shift(k) for k in range(1, p + 1)}, axis=1)
    mask = lags.notna().all(axis=1)
    return lags[mask], s[mask], s


def acf(s: np.ndarray, nlags: int) -> np.ndarray:
    s = s - s.mean()
    denom = float(s @ s) or 1.0
    return np.array([1.0] + [float(s[k:] @ s[:-k]) / denom for k in range(1, nlags + 1)])


def ols_fit(X: np.ndarray, y: np.ndarray):
    """公式解 (Normal Equation): beta = (X'X)^+ X'y, 回傳 (b0, w)"""
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.pinv(A.T @ A) @ A.T @ y
    return beta[0], beta[1:]


def gd_fit(X: np.ndarray, y: np.ndarray, lr: float, iters: int, scale: bool):
    """梯度下降法; scale=True 時對 X, y 做標準化後再還原回原始單位"""
    n, p = X.shape
    if scale:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        my, sy = y.mean(), (y.std() or 1.0)
        Xs, ys = (X - mu) / sd, (y - my) / sy
    else:
        Xs, ys, mu, sd, my, sy = X, y, np.zeros(p), np.ones(p), 0.0, 1.0
    w, b, losses = np.zeros(p), 0.0, []
    for _ in range(int(iters)):
        err = Xs @ w + b - ys
        loss = float(np.mean(err ** 2))
        if not np.isfinite(loss) or loss > 1e30:
            losses.append(np.inf)
            return np.nan, np.full(p, np.nan), losses
        losses.append(loss)
        w -= lr * 2 / n * (Xs.T @ err)
        b -= lr * 2 * err.mean()
    if scale:
        w_o = w * sy / sd
        b_o = my + sy * b - w_o @ mu
        return b_o, w_o, losses
    return b, w, losses


def metrics(y, yhat, p):
    """以 sklearn.metrics 計算 R²、Adj R²、RMSE、MAE"""
    n = len(y)
    r2 = r2_score(y, yhat)
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan
    return {"R²": r2, "Adj R²": adj, "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
            "MAE": float(mean_absolute_error(y, yhat))}


def coef_table(X: np.ndarray, y: np.ndarray, names: list[str], b0, w):
    """係數、標準誤、t、p-value (OLS 假設)"""
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    beta = np.r_[b0, w]
    resid = y - A @ beta
    dof = max(n - int(np.linalg.matrix_rank(A)), 1)  # 完整 One-Hot 與截距共線時, 以矩陣秩計算自由度
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.abs(np.diag(np.linalg.pinv(A.T @ A))) * sigma2)
    t = beta / np.where(se == 0, np.nan, se)
    pv = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    return pd.DataFrame({"變數": ["截距 (Intercept)"] + names, "係數": beta, "標準誤": se, "t 值": t, "p-value": pv})


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R²_j), R²_j 為第 j 個特徵對其他特徵迴歸的 R²。完全共線 (如完整 One-Hot) 時為 inf。"""
    p = X.shape[1]
    if p < 2:
        return pd.DataFrame({"變數": X.columns, "VIF": [1.0] * p})
    V, out = X.values.astype(float), []
    for j in range(p):
        others = np.column_stack([np.ones(len(V)), np.delete(V, j, axis=1)])
        res = V[:, j] - others @ np.linalg.lstsq(others, V[:, j], rcond=None)[0]
        tot = float(((V[:, j] - V[:, j].mean()) ** 2).sum())
        r2 = 1 - float(res @ res) / tot if tot > 0 else 0.0
        out.append(np.inf if r2 > 1 - 1e-9 else 1.0 / (1.0 - r2))
    return pd.DataFrame({"變數": X.columns, "VIF": out})


# ------------------------------------------------------------------
# 特徵篩選 (Feature Selection) - Filter / Wrapper / Embedded (sklearn)
# ------------------------------------------------------------------
FS_CATEGORIES = {
    "不篩選": {"不篩選 (使用全部特徵)": "none"},
    "Filter 過濾法": {
        "相關係數過濾 (|r| ≥ 門檻)": "corr",
        "SelectKBest - F 檢定": "kbest",
        "SelectKBest - 互資訊 (Mutual Info)": "kbest_mi",
    },
    "Wrapper 包裝法": {
        "RFE 遞迴特徵消除": "rfe",
        "RFECV (交叉驗證自動決定特徵數)": "rfecv",
        "前向選擇 (SFS forward)": "sfs",
        "後向選擇 (SFS backward)": "sfs_bwd",
        "向後逐步消除 (p-value)": "backward",
    },
    "Embedded 嵌入法": {
        "LassoCV (L1 正則化)": "lasso",
        "ElasticNetCV (L1+L2 正則化)": "enet",
        "隨機森林特徵重要度 (SelectFromModel)": "rf",
    },
}
FS_METHODS = {n: m for d in FS_CATEGORIES.values() for n, m in d.items()}
FS_NAME_OF = {m: n for n, m in FS_METHODS.items()}
FS_CAT_OF = {m: c for c, d in FS_CATEGORIES.items() for m in d.values()}
K_BASED = {"kbest", "kbest_mi", "rfe", "sfs", "sfs_bwd"}  # 需要指定保留特徵數 k 的方法
FS_DESC = {
    "Filter 過濾法": "不依賴任何模型, 只用統計量 (相關係數、F 檢定、互資訊) 替每個特徵打分數後篩選。速度最快, 但看不到特徵間的交互作用。",
    "Wrapper 包裝法": "把『模型表現』當作評分標準, 反覆訓練模型來挑選特徵子集 (RFE、SFS、逐步消除)。通常較準, 但計算量較大。",
    "Embedded 嵌入法": "在模型訓練的過程中同時完成篩選 (Lasso 把不重要的係數壓成 0、隨機森林提供重要度)。兼顧效率與準確度。",
    "不篩選": "使用全部特徵, 作為比較基準 (baseline)。",
}


def _cv_n(n: int) -> int:
    return min(5, max(2, n // 5))


def _abs_corr(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    out = []
    for c in X.columns:
        v = X[c].values.astype(float)
        out.append(abs(np.corrcoef(v, y)[0, 1]) if v.std() > 0 else 0.0)
    return np.nan_to_num(np.array(out))


@st.cache_data(show_spinner=False)
def feature_diagnostics(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """各特徵診斷指標: Filter (|r|、F、p-value、互資訊、VIF) / Wrapper (RFE 排名) / Embedded (Lasso 係數、RF 重要度)"""
    cols = list(X.columns)
    Xv = X.values.astype(float)
    Xs = StandardScaler().fit_transform(Xv)
    F, pv = f_regression(Xv, y)
    mi = mutual_info_regression(Xv, y, random_state=0)
    lasso = LassoCV(cv=_cv_n(len(y)), max_iter=20000, random_state=0).fit(Xs, y)
    rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(Xv, y)
    rank = RFE(LinearRegression(), n_features_to_select=1).fit(Xs, y).ranking_ if len(cols) > 1 else [1]
    return pd.DataFrame({
        "特徵": cols, "|r| 與 y": _abs_corr(X, y), "F 值": np.nan_to_num(F), "p-value": np.nan_to_num(pv, nan=1.0),
        "互資訊": mi, "VIF": vif_table(X)["VIF"].values,
        "RFE 排名": rank, "Lasso 係數": lasso.coef_, "RF 重要度": rf.feature_importances_,
    })


def _rank_order(kind: str, X: pd.DataFrame, y: np.ndarray, Xs: np.ndarray, cvn: int) -> list:
    """把『不指定 k』的方法轉成特徵排序 (由重要到不重要), 供『特徵數量 vs 表現』圖依序取前 k 個。
    lasso / enet: |標準化係數| 大小; rf: 特徵重要度; corr: |r|; rfe: 標準化係數遞迴消除的順序。"""
    p = X.shape[1]
    tie = _abs_corr(X, y)  # 係數相同 (例如 Lasso 壓成 0) 時, 以 |r| 決定先後
    if kind in ("lasso", "enet"):
        mdl = (LassoCV(cv=cvn, max_iter=20000, random_state=0) if kind == "lasso"
               else ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 1.0], cv=cvn, max_iter=20000, random_state=0)).fit(Xs, y)
        score = np.abs(mdl.coef_)
    elif kind == "rf":
        score = RandomForestRegressor(n_estimators=100, random_state=0).fit(X.values.astype(float), y).feature_importances_
    elif kind == "corr":
        score = tie
    else:  # rfe
        rk = RFE(LinearRegression(), n_features_to_select=1).fit(Xs, y).ranking_
        score = -np.asarray(rk, dtype=float)
    return [int(i) for i in sorted(range(p), key=lambda i: (-score[i], -tie[i]))]


def select_features(method: str, X: pd.DataFrame, y: np.ndarray, k: int, c_thr: float, p_thr: float):
    """回傳 (被選中的特徵名稱, 逐步消除紀錄)。篩選只使用傳入的訓練資料, 避免資料洩漏。"""
    cols, p = list(X.columns), X.shape[1]
    k = max(1, min(k, p))
    log = []
    if method == "none" or p < 2 or (method in K_BASED and k >= p):
        return cols, log
    Xv = X.values.astype(float)
    Xs = StandardScaler().fit_transform(Xv)
    cvn = _cv_n(len(y))
    if method == "corr":
        r = _abs_corr(X, y)
        sel = [c for c, v in zip(cols, r) if v >= c_thr] or [cols[int(np.argmax(r))]]
    elif method == "kbest":
        sel = [c for c, m in zip(cols, SelectKBest(f_regression, k=k).fit(Xv, y).get_support()) if m]
    elif method == "kbest_mi":
        sk = SelectKBest(partial(mutual_info_regression, random_state=0), k=k).fit(Xv, y)
        sel = [c for c, m in zip(cols, sk.get_support()) if m]
    elif method == "rfe":
        sel = [c for c, m in zip(cols, RFE(LinearRegression(), n_features_to_select=k).fit(Xs, y).support_) if m]
    elif method == "rfecv":
        sel = [c for c, m in zip(cols, RFECV(LinearRegression(), cv=cvn, min_features_to_select=1,
                                             scoring="neg_root_mean_squared_error").fit(Xs, y).support_) if m]
    elif method in ("sfs", "sfs_bwd"):
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, cv=cvn,
                                        direction="forward" if method == "sfs" else "backward").fit(Xs, y)
        sel = [c for c, m in zip(cols, sfs.get_support()) if m]
    elif method == "backward":
        sel = list(cols)
        while len(sel) > 1:
            sub = X[sel].values
            # 完整 One-Hot 與截距共線時 p-value 無法估計: 先把線性相依的欄位當作『基準類別』移除 (紀錄 p=1.0)
            A = np.column_stack([np.ones(len(sub)), sub])
            keep = [0]
            for j in range(1, A.shape[1]):
                if np.linalg.matrix_rank(A[:, keep + [j]]) > len(keep):
                    keep.append(j)
            dep = [sel[j] for j in range(len(sel)) if (j + 1) not in keep]
            if dep:
                log.append((dep[0], 1.0))
                sel.remove(dep[0])
                continue
            b0, w0 = ols_fit(sub, y)
            ct = coef_table(sub, y, sel, b0, w0).iloc[1:].reset_index(drop=True)
            ct["p-value"] = ct["p-value"].fillna(1.0)
            worst = ct.loc[ct["p-value"].idxmax()]
            if worst["p-value"] > p_thr:
                log.append((worst["變數"], float(worst["p-value"])))
                sel.remove(worst["變數"])
            else:
                break
    elif method.startswith("rank:"):
        sel = [cols[i] for i in _rank_order(method[5:], X, y, Xs, cvn)[:k]]
    elif method in ("lasso", "enet"):
        mdl = (LassoCV(cv=cvn, max_iter=20000, random_state=0) if method == "lasso"
               else ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 1.0], cv=cvn, max_iter=20000, random_state=0)).fit(Xs, y)
        sel = [c for c, w in zip(cols, mdl.coef_) if abs(w) > 1e-8] or [cols[int(np.argmax(np.abs(mdl.coef_)))]]
    else:  # rf
        sfm = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=0), threshold="mean").fit(Xv, y)
        sup = sfm.get_support()
        sel = [c for c, m in zip(cols, sup) if m] or [cols[int(np.argmax(sfm.estimator_.feature_importances_))]]
    return sel, log


class FeatureSelector(BaseEstimator, TransformerMixin):
    """把 select_features 包成 sklearn 轉換器, 方便放進 Pipeline 做『折內篩選』的巢狀交叉驗證 (無資料洩漏)"""

    def __init__(self, method="none", k=3, c_thr=0.1, p_thr=0.05, allowed=None):
        self.method, self.k, self.c_thr, self.p_thr, self.allowed = method, k, c_thr, p_thr, allowed

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        idx = list(self.allowed) if self.allowed is not None else list(range(X.shape[1]))
        df = pd.DataFrame(X[:, idx], columns=[str(i) for i in idx])
        sel, _ = select_features(self.method, df, np.asarray(y, float), self.k, self.c_thr, self.p_thr)
        self.support_ = [int(s) for s in sel]
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.support_]


# ------------------------------------------------------------------
# 最佳模型搜尋 (Model Selection)
# ------------------------------------------------------------------
def _rmse(y, yh):
    return float(np.sqrt(mean_squared_error(y, yh)))


def _candidates(p: int, k: int, search_k: bool, num_idx):
    """候選模型 = (顯示名稱, 分類, 方法, k, 允許的欄位)。皆以 LinearRegression 作最終估計器。"""
    kk = max(1, min(k, p - 1))
    ks = sorted({kk} | ({1, 2, 3, 5, 8} if search_k else set()))
    ks = [x for x in ks if 1 <= x <= p - 1]
    C = [("全部特徵 (不篩選)", "不篩選", "none", p, None)]
    if num_idx:
        C.append(("單純線性回歸 (F 檢定選 1 個數值特徵)", "Filter 過濾法", "kbest", 1, tuple(num_idx)))
    C.append((FS_NAME_OF["corr"], "Filter 過濾法", "corr", kk, None))
    for x in ks:
        C.append((f"SelectKBest-F (k={x})", "Filter 過濾法", "kbest", x, None))
    for x in ks:
        C.append((f"SelectKBest-互資訊 (k={x})", "Filter 過濾法", "kbest_mi", x, None))
    for x in ks:
        C.append((f"RFE (k={x})", "Wrapper 包裝法", "rfe", x, None))
    C.append((FS_NAME_OF["rfecv"], "Wrapper 包裝法", "rfecv", kk, None))
    C.append((f"前向選擇 SFS (k={kk})", "Wrapper 包裝法", "sfs", kk, None))
    C.append((f"後向選擇 SFS (k={kk})", "Wrapper 包裝法", "sfs_bwd", kk, None))
    C.append((FS_NAME_OF["backward"], "Wrapper 包裝法", "backward", kk, None))
    for m in ("lasso", "enet", "rf"):
        C.append((FS_NAME_OF[m], "Embedded 嵌入法", m, kk, None))
    return C


@st.cache_data(show_spinner="搜尋最佳模型中 (折內篩選的巢狀交叉驗證)...")
def search_regression(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, yte: np.ndarray,
                      seed: int, k: int, c_thr: float, p_thr: float, search_k: bool, num_idx: tuple) -> pd.DataFrame:
    """每個候選: 5 折 CV 的 RMSE / R² (特徵篩選在每折的訓練部分內重做, 避免資料洩漏) + 測試集表現。"""
    kf = KFold(5, shuffle=True, random_state=int(seed))
    rows = []
    for label, cat, m, kk, allowed in _candidates(Xtr.shape[1], k, search_k, list(num_idx)):
        try:
            pipe = Pipeline([("sel", FeatureSelector(m, kk, c_thr, p_thr, allowed)), ("lr", LinearRegression())])
            cv = cross_validate(pipe, Xtr.values.astype(float), ytr, cv=kf,
                                scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
            pipe.fit(Xtr.values.astype(float), ytr)
            sel = [Xtr.columns[i] for i in pipe.named_steps["sel"].support_]
            yh = pipe.predict(Xte.values.astype(float))
        except Exception:
            continue
        rows.append({"模型": label, "分類": cat, "method": m, "k": int(kk), "特徵數": len(sel), "support": sel,
                     "保留特徵": ", ".join(sel), "cv_rmse": float(-cv["test_rmse"].mean()),
                     "cv_std": float(cv["test_rmse"].std()), "cv_r2": float(cv["test_r2"].mean()),
                     "test_rmse": _rmse(yte, yh), "test_r2": float(r2_score(yte, yh))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="比較各單一特徵中...")
def search_simple(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, yte: np.ndarray, seed: int) -> pd.DataFrame:
    """單純線性回歸: 每個數值特徵各建一個模型, 以 5 折 CV RMSE 比較。"""
    kf = KFold(5, shuffle=True, random_state=int(seed))
    rows = []
    for c in Xtr.columns:
        a, b = Xtr[[c]].values.astype(float), Xte[[c]].values.astype(float)
        cv = cross_validate(LinearRegression(), a, ytr, cv=kf, scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
        yh = LinearRegression().fit(a, ytr).predict(b)
        rows.append({"模型": f"單純線性回歸: {c}", "分類": "", "method": "", "k": 1, "特徵數": 1, "support": [c],
                     "保留特徵": c, "cv_rmse": float(-cv["test_rmse"].mean()), "cv_std": float(cv["test_rmse"].std()),
                     "cv_r2": float(cv["test_r2"].mean()), "test_rmse": _rmse(yte, yh), "test_r2": float(r2_score(yte, yh))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="搜尋最佳 AR 階數中...")
def search_ar(series: np.ndarray, pmax: int, test_size: float) -> pd.DataFrame:
    """候選: 樸素預測 y(t)=y(t-1) 與 AR(1..pmax); 全部使用相同樣本列、依時間順序切分與 TimeSeriesSplit。"""
    s = pd.Series(series)
    lags = pd.concat({i: s.shift(i) for i in range(1, pmax + 1)}, axis=1)
    mask = lags.notna().all(axis=1)
    L, yv = lags[mask].values, s[mask].values
    n_tr = int(round(len(yv) * (1 - test_size)))
    Ltr, ytr, Lte, yte = L[:n_tr], yv[:n_tr], L[n_tr:], yv[n_tr:]
    rows = []
    for p in range(0, pmax + 1):
        rm, r2s = [], []
        for a, b in TimeSeriesSplit(5).split(Ltr):
            yh = Ltr[b, 0] if p == 0 else LinearRegression().fit(Ltr[a, :p], ytr[a]).predict(Ltr[b, :p])
            rm.append(_rmse(ytr[b], yh)); r2s.append(r2_score(ytr[b], yh))
        yh = Lte[:, 0] if p == 0 else LinearRegression().fit(Ltr[:, :p], ytr).predict(Lte[:, :p])
        rows.append({"模型": "樸素預測 y(t)=y(t-1)" if p == 0 else f"AR({p})", "分類": "", "method": "", "k": p,
                     "特徵數": p, "support": [f"y(t-{i})" for i in range(1, p + 1)],
                     "保留特徵": "" if p == 0 else ", ".join(f"y(t-{i})" for i in range(1, p + 1)),
                     "cv_rmse": float(np.mean(rm)), "cv_std": float(np.std(rm)), "cv_r2": float(np.mean(r2s)),
                     "test_rmse": _rmse(yte, yh), "test_r2": float(r2_score(yte, yh))})
    return pd.DataFrame(rows)


def pick_best(df: pd.DataFrame):
    """回傳 (最低 CV RMSE 的模型, 建議模型)。建議模型採 1-SE 規則: CV RMSE 在最佳者 1 個標準誤內、且特徵數最少。"""
    best = df.loc[df["cv_rmse"].idxmin()]
    ok = df[df["cv_rmse"] <= best["cv_rmse"] + best["cv_std"] / np.sqrt(5)]
    return best, ok.sort_values(["特徵數", "cv_rmse"]).iloc[0]


@st.cache_data(show_spinner="計算不同特徵數量的表現...")
def sweep_feature_count(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, yte: np.ndarray,
                        method: str, seed: int, c_thr: float, p_thr: float) -> pd.DataFrame:
    """特徵數量 k = 1..p: 用指定篩選方法挑 k 個特徵 → 測試集 RMSE / R²; 另附折內篩選的 5 折 CV RMSE。"""
    p = Xtr.shape[1]
    kf = KFold(5, shuffle=True, random_state=int(seed))
    rows = []
    for kk in range(1, p + 1):
        sel, _ = select_features(method, Xtr, ytr, kk, c_thr, p_thr)
        yh = LinearRegression().fit(Xtr[sel].values, ytr).predict(Xte[sel].values)
        pipe = Pipeline([("sel", FeatureSelector(method, kk, c_thr, p_thr)), ("lr", LinearRegression())])
        cv = cross_validate(pipe, Xtr.values.astype(float), ytr, cv=kf, scoring="neg_root_mean_squared_error")
        rows.append({"特徵數": kk, "保留特徵": "[" + ", ".join(sel) + "]", "RMSE": _rmse(yte, yh),
                     "R²": float(r2_score(yte, yh)), "CV RMSE (訓練集)": float(-cv["test_score"].mean()),
                     "CV 標準差": float(cv["test_score"].std())})
    return pd.DataFrame(rows)


def count_chart(df: pd.DataFrame, xcol: str, xlabel: str, kind: str = "line"):
    """左: RMSE by Number of Features / 右: R-squared by Number of Features"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"RMSE by {xlabel}", f"R-squared by {xlabel}"))
    mode = "lines+markers"
    fig.add_trace(go.Scatter(x=df[xcol], y=df["RMSE"], mode=mode, name="測試集 RMSE", line=dict(color="#3b78b5")), 1, 1)
    if "CV RMSE (訓練集)" in df:
        fig.add_trace(go.Scatter(x=df[xcol], y=df["CV RMSE (訓練集)"], mode=mode, name="CV RMSE (訓練集)",
                                 line=dict(color="#e08a1e", dash="dash"), visible="legendonly"), 1, 1)
    fig.add_trace(go.Scatter(x=df[xcol], y=df["R²"], mode=mode, name="測試集 R²", line=dict(color="#3b78b5"),
                             showlegend=False), 1, 2)
    i_rmse, i_r2 = int(df["RMSE"].idxmin()), int(df["R²"].idxmax())
    fig.add_trace(go.Scatter(x=[df.loc[i_rmse, xcol]], y=[df.loc[i_rmse, "RMSE"]], mode="markers", name="RMSE 最低",
                             marker=dict(color="crimson", size=13, symbol="star"), visible="legendonly"), 1, 1)
    fig.add_trace(go.Scatter(x=[df.loc[i_r2, xcol]], y=[df.loc[i_r2, "R²"]], mode="markers", name="R² 最高",
                             marker=dict(color="crimson", size=13, symbol="star"), showlegend=False), 1, 2)
    fig.update_xaxes(title_text=xlabel, dtick=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="R-squared", row=1, col=2)
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.25))
    return fig


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("📂 問題與資料來源")
problem_name = st.sidebar.selectbox("選擇問題 (下拉選單)", list(PROBLEMS.keys()))
problem = PROBLEMS[problem_name]

MODELS = {
    "多元線性回歸 (Multiple Linear Regression)": "multiple",
    "單純線性回歸 (Simple Linear Regression)": "simple",
    "自迴歸 (Auto Regression, AR)": "ar",
}
st.sidebar.header("🧠 模型選擇")
model_name = st.sidebar.selectbox("選擇迴歸模型", list(MODELS.keys()))
model_kind = MODELS[model_name]
MODEL_SHORT = model_name.split(" (")[0]

with st.sidebar.expander("📂 資料來源 (自訂 URL / 上傳 CSV)", expanded=False):
    custom_url = st.text_input(
        "自訂 Dataset URL (可任意輸入, 留空則使用上方內建資料)",
        value="",
        placeholder="https://raw.githubusercontent.com/.../xxx.csv",
    )
    header_none = st.checkbox("資料沒有標題列 (header=None)", value=False)
    uploaded = st.file_uploader("或上傳 CSV 檔", type=["csv", "txt"])

st.sidebar.header("⚙️ 資料切分")
seed = st.sidebar.number_input("隨機數種子 (Seed)", 0, 99999, 0)
test_size = st.sidebar.slider("測試集比例", 0.1, 0.5, 0.2, 0.05)
n_outliers = st.sidebar.slider("標記離群值個數", 1, 20, 10)

with st.sidebar.expander("📉 梯度下降設定", expanded=False):
    lr = st.number_input("學習率 (Learning Rate)", 0.00001, 1.0, 0.05, step=0.01, format="%.5f")
    iters = st.slider("迭代次數 (Iterations)", 100, 20000, 2000, 100)
    use_scaling = st.checkbox("啟用特徵縮放 (Feature Scaling)", True,
                              help="GD 對特徵尺度很敏感, 建議開啟")

# ------------------------------------------------------------------
# 載入資料
# ------------------------------------------------------------------
try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded, sep=None, engine="python",
                          header=None if header_none else "infer")
        if header_none:
            raw.columns = [f"x{i}" for i in range(raw.shape[1] - 1)] + ["y"]
        source, is_custom = f"上傳檔案 {uploaded.name}", True
    elif custom_url.strip():
        raw = read_csv_url(custom_url.strip(), header_none)
        source, is_custom = custom_url.strip(), True
    else:
        raw, source = load_builtin(problem["key"])
        is_custom = False
except Exception as e:
    st.error(f"❌ 無法讀取資料: {e}")
    st.stop()

raw.columns = [str(c).strip() for c in raw.columns]
cols = list(raw.columns)
if len(cols) < 2:
    st.error("資料至少需要 2 個欄位 (1 個目標 + 1 個特徵)。")
    st.stop()

st.sidebar.header("🎯 變數選擇")
default_target = cols[-1] if is_custom else next((c for c in cols if c.lower() == problem["target"].lower()), cols[-1])
target = st.sidebar.selectbox("目標變數 (y)", cols, index=cols.index(default_target))
ar_p = 1
board_simple = board_ar = None
auto_ar_note = ""
if model_kind == "multiple":
    default_feats = [c for c in cols if c != target]
    if not is_custom and problem["key"] == "startups":  # 老師的設定: 不使用 Administration
        default_feats = [c for c in default_feats if c != "Administration"]
    with st.sidebar.expander("特徵變數 (X)" + (" - 預設排除 Administration" if len(default_feats) < len(cols) - 1 else " - 預設全選"),
                             expanded=False):
        feats = st.multiselect("特徵變數 (X)", [c for c in cols if c != target],
                               default=default_feats, label_visibility="collapsed")
elif model_kind == "simple":
    y_num = pd.to_numeric(raw[target], errors="coerce")
    num_cols = [c for c in cols if c != target and pd.api.types.is_numeric_dtype(raw[c])]
    if not num_cols:
        st.error("單純線性回歸需要至少一個數值型特徵欄位。")
        st.stop()
    # 每個數值特徵各建一個單純線性回歸, 只用訓練列做 5 折 CV (切分與下方主流程一致)
    try:
        valid = y_num.notna().values
        Xc = raw.loc[valid, num_cols].apply(pd.to_numeric, errors="coerce")
        Xc = Xc.fillna(Xc.median()).reset_index(drop=True)
        yc = y_num[valid].values
        i_tr, i_te = train_test_split(np.arange(len(yc)), test_size=test_size, random_state=int(seed))
        board_simple = search_simple(Xc.iloc[i_tr], yc[i_tr], Xc.iloc[i_te], yc[i_te], int(seed))
    except Exception:
        board_simple = None
    corr_abs = {c: abs(raw[c].corr(y_num)) for c in num_cols}
    best = max(corr_abs, key=lambda c: -1 if np.isnan(corr_abs[c]) else corr_abs[c])
    pick_mode = st.sidebar.radio("單一特徵選擇方式", ["手動選擇", "🏆 自動 (CV RMSE 最低)"],
                                 disabled=board_simple is None)
    if pick_mode.startswith("🏆") and board_simple is not None:
        best = board_simple.loc[board_simple["cv_rmse"].idxmin(), "保留特徵"]
        st.sidebar.success(f"自動選定特徵: {best}")
        feats = [best]
    else:
        feats = [st.sidebar.selectbox("特徵變數 (單一 X)", num_cols, index=num_cols.index(best),
                                      help="預設為與目標變數相關性最高的欄位")]
else:  # ar
    feats = []
    s_raw = pd.to_numeric(raw[target], errors="coerce").dropna().reset_index(drop=True).astype(float)
    pmax = int(min(20, max(2, len(s_raw) // 5)))
    try:
        board_ar = search_ar(s_raw.values, pmax, test_size)
    except Exception:
        board_ar = None
    ar_mode = st.sidebar.radio("自迴歸階數 p 選擇方式", ["手動選擇", "🏆 自動 (CV RMSE 最低)", "🏆 自動 (建議: 1-SE 最簡)"],
                               disabled=board_ar is None)
    if ar_mode != "手動選擇" and board_ar is not None:
        b_ar, s_ar = pick_best(board_ar[board_ar["k"] >= 1])
        ar_p = int((b_ar if "最低" in ar_mode else s_ar)["k"])
        st.sidebar.success(f"自動選定 AR({ar_p})")
        auto_ar_note = f"已自動採用 AR({ar_p})"
    else:
        ar_p = st.sidebar.slider("自迴歸階數 p (使用前 p 期的 y)", 1, pmax, min(3, pmax))
    st.sidebar.caption("AR 以資料列順序視為時間序列, 並依時間順序切分訓練/測試集 (不隨機打亂)。")
if model_kind != "ar" and not feats:
    st.warning("請至少選擇一個特徵變數。")
    st.stop()

onehot_base = {}  # 每個類別變數被移除的基準類別
onehot_drop = True  # True: 移除一個基準類別 (n-1 欄, 避免虛擬變數陷阱); False: 完整 One-Hot (n 欄)
if model_kind == "multiple":
    cat_feats = [c for c in feats if not pd.api.types.is_numeric_dtype(raw[c])]
    if cat_feats:
        oh_opts = ["完整 One-Hot (每個類別一欄, 如 100 / 010 / 001)", "移除一個基準類別 (n−1 欄, 避免虛擬變數陷阱)"]
        oh_choice = ctrl_box.radio(f"類別特徵 ({', '.join(cat_feats)}) 的 One-Hot 編碼方式", oh_opts, horizontal=True,
                                   help="50 Startups 的 State 有 3 個州。完整 One-Hot: 每州一欄 (State_California / Florida / New York); "
                                        "老師的 5 個特徵 = R&D Spend + Marketing Spend + 3 個州別欄。"
                                        "移除基準類別: 只留 2 個州別欄, 另一州以 00 表示。")
        onehot_drop = oh_choice == oh_opts[1]
        if onehot_drop:
            bcols = ctrl_box.columns(len(cat_feats))
            for bc, c in zip(bcols, cat_feats):
                lv = sorted(raw[c].dropna().astype(str).unique())
                onehot_base[c] = bc.selectbox(f"{c} 的基準類別 (被移除, 以全 0 表示)", lv, index=0)
try:
    if model_kind == "ar":
        X_df, y_s, series_all = prepare_ar(raw, target, ar_p)
        cat_cols, clean = [], pd.DataFrame({target: series_all})
    else:
        X_df, y_s, cat_cols, clean = prepare_xy(raw, target, feats, onehot_drop, onehot_base)
except Exception as e:
    st.error(f"❌ 資料準備失敗: {e}")
    st.stop()
if len(X_df) < X_df.shape[1] + 5:
    st.error("有效資料筆數太少, 無法建立模型。")
    st.stop()

# ---- 訓練/測試切分 (先切分, 再只用訓練集做特徵篩選, 避免資料洩漏) ----
n_all = len(X_df)
if model_kind == "ar":  # 時間序列: 依時間順序切分
    n_tr = int(round(n_all * (1 - test_size)))
    idx_tr, idx_te = np.arange(n_tr), np.arange(n_tr, n_all)
else:
    idx_tr, idx_te = train_test_split(np.arange(n_all), test_size=test_size, random_state=int(seed))
if len(idx_te) < 2 or len(idx_tr) < X_df.shape[1] + 2:
    st.error("訓練/測試資料太少, 請調整測試集比例或階數 p。")
    st.stop()
y_all = y_s.values
y_tr, y_te = y_all[idx_tr], y_all[idx_te]
X_full_df = X_df  # 編碼後的全部候選特徵

# ---- 特徵篩選 (Filter / Wrapper / Embedded) 與 最佳模型選擇 ----
fs_method, fs_log, fs_name = "none", [], list(FS_METHODS)[0]
k_keep, c_thr, p_thr = 1, 0.1, 0.05
board_reg, best_reg, sugg_reg, chosen_reg, search_k = None, None, None, None, False
if model_kind == "multiple" and X_full_df.shape[1] >= 2:
    p_enc = X_full_df.shape[1]
    with ctrl_box:
        with st.expander("🔎 特徵篩選與 🏆 最佳模型選擇 (Filter / Wrapper / Embedded)", expanded=True):
            st.markdown("**① 選擇特徵篩選演算法**")
            c_a, c_b = st.columns([1, 2])
            fs_cat = c_a.selectbox("篩選類別", list(FS_CATEGORIES.keys()), index=3)
            m_names = list(FS_CATEGORIES[fs_cat].keys())
            fs_name = c_b.selectbox("篩選方法", m_names, index=0)
            fs_method = FS_METHODS[fs_name]
            st.caption(FS_DESC[fs_cat])
            st.markdown("**② 參數**")
            c_k, c_c, c_p = st.columns(3)
            k_keep = c_k.slider("保留特徵數 k (KBest / RFE / SFS)", 1, p_enc, min(5, p_enc))
            c_thr = c_c.slider("相關係數門檻 |r| (相關係數過濾)", 0.0, 0.9, 0.1, 0.05)
            p_thr = c_p.slider("p-value 門檻 (向後逐步消除)", 0.01, 0.30, 0.05, 0.01)
            st.markdown("**③ 最佳模型選擇**")
            c_r, c_s = st.columns(2)
            run_search = c_r.checkbox("執行最佳模型搜尋", True,
                                      help="每個候選模型都做 5 折 CV, 且特徵篩選在每折內重做 (無資料洩漏)。首次執行較耗時, 之後有快取。")
            search_k = c_s.checkbox("同時搜尋最佳特徵數 k", False, disabled=not run_search)
            adopt = st.radio("模型採用方式", ["手動 (使用上方所選方法)", "🏆 自動採用最佳模型 (CV RMSE 最低)",
                                            "🏆 自動採用建議模型 (1-SE, 最精簡)"], horizontal=True, disabled=not run_search)
    if run_search:
        num_enc = [i for i, c in enumerate(X_full_df.columns) if c in feats and c not in cat_cols]
        try:
            board_reg = search_regression(X_full_df.iloc[idx_tr], y_tr, X_full_df.iloc[idx_te], y_te, int(seed),
                                          int(k_keep), float(c_thr), float(p_thr), bool(search_k), tuple(num_enc))
            best_reg, sugg_reg = pick_best(board_reg)
        except Exception as e:
            st.sidebar.warning(f"最佳模型搜尋失敗: {e}")
        if board_reg is not None and not adopt.startswith("手動"):
            chosen_reg = best_reg if "最低" in adopt else sugg_reg
if chosen_reg is not None:
    selected = list(chosen_reg["support"])
    fs_method, fs_name, k_keep = chosen_reg["method"], f"🏆 自動採用: {chosen_reg['模型']}", int(chosen_reg["k"])
    if fs_method == "backward":
        fs_log = select_features("backward", X_full_df.iloc[idx_tr], y_tr, k_keep, c_thr, p_thr)[1]
else:
    try:
        selected, fs_log = select_features(fs_method, X_full_df.iloc[idx_tr], y_tr, k_keep, c_thr, p_thr)
    except Exception as e:
        st.warning(f"特徵篩選失敗 ({e}), 改用全部特徵。")
        selected, fs_log = list(X_full_df.columns), []
X_df = X_full_df[selected]
names = list(selected)
X_all = X_df.values
X_tr, X_te = X_all[idx_tr], X_all[idx_te]

# 建模 (sklearn LinearRegression 為主要模型; 手刻公式解與梯度下降作為對照)
sk_model = LinearRegression().fit(X_tr, y_tr)
b_sk, w_sk = float(sk_model.intercept_), sk_model.coef_
b_ols, w_ols = ols_fit(X_tr, y_tr)
b_gd, w_gd, losses = gd_fit(X_tr, y_tr, lr, iters, use_scaling)
gd_ok = bool(np.isfinite(b_gd))
pred = lambda X, b, w: X @ w + b  # noqa: E731

# ------------------------------------------------------------------
# 標題與 CRISP-DM 分頁
# ------------------------------------------------------------------
with head_box:
    st.title(f"🟪 利用 CRISP-DM 架構 - {MODEL_SHORT}互動網頁")
    st.caption(f"使用 OLS 公式解與梯度下降法 (Gradient Descent) 建立「{model_name}」模型  |  "
               f"目前問題: **{'自訂資料集' if is_custom else problem_name}**")

tabs = st.tabs(["1. 商業理解", "2. 數據理解", "3. 數據準備", "4. 建立模型", "5. 模型評估", "6. 模型部署"])

# ---------------- 1. 商業理解 ----------------
with tabs[0]:
    st.header("🏢 第一階段: 商業理解 (Business Understanding)")
    st.write("在 CRISP-DM 流程中, 第一步是**確立商業目標**, 並把它轉化為明確的資料探勘目標。")
    c1, c2 = st.columns(2)
    if is_custom:
        biz = f"您載入了自訂資料集, 想利用 **{len(feats)} 個特徵** 預測 **{target}**。"
        qs = [f"哪些特徵對 {target} 有顯著影響?", f"模型能解釋 {target} 多少變異 (R²)?",
              "公式解與梯度下降法的結果是否一致?"]
    elif problem["key"] == "startups":
        biz = ("您是創投公司的分析師, 面對的挑戰是: **「如何評估一家新創公司的獲利能力, 並決定資金該投入哪些面向 "
               "(研發、行政、行銷)?」**")
        qs = ["R&D、行政、行銷支出與利潤之間是否存在線性關係?", "哪一項支出對利潤的邊際貢獻最大?",
              "公司所在州別是否對利潤有顯著影響?"]
    else:
        biz = ("您是房地產顧問, 面對的挑戰是: **「如何依據社區特性 (犯罪率、房間數、低收入人口比例等) 合理估計波士頓地區的房價中位數?」**")
        qs = ["哪些社區特徵 (如 RM、LSTAT、PTRATIO) 對房價影響最大?", "犯罪率或空氣污染 (NOX) 每上升一單位, 房價約下降多少?",
              "在給定社區條件下, 預期房價是多少?"]
    if model_kind == "simple":
        biz += f"\n\n本次採用**單純線性回歸**, 只使用單一特徵 **{feats[0]}** 來解釋 **{target}**。"
        qs = [f"{feats[0]} 與 {target} 之間是否存在明顯的線性關係?",
              f"{feats[0]} 每增加 1 單位, {target} 平均變化多少 (斜率)?",
              f"給定 {feats[0]}, 預期的 {target} 是多少?"]
    elif model_kind == "ar":
        biz = (f"您想利用 **{target}** 過去的表現預測下一期的數值 (自迴歸: 以自己的歷史值預測自己)。"
               f"本次使用前 **{ar_p}** 期資料建立 AR({ar_p}) 模型。")
        qs = [f"{target} 是否具有自我相關 (前期數值能預測後期)?", f"AR({ar_p}) 對測試期的預測誤差有多大?",
              "未來數期的遞迴預測趨勢為何?"]
    with c1:
        with st.container(border=True):
            st.subheader("🎯 商業場景設定")
            st.write(biz)
            st.write("需要回答的問題:")
            for q in qs:
                st.markdown(f"- {q}")
    with c2:
        with st.container(border=True):
            st.subheader("🧠 數據挖掘目標 (Data Goal)")
            if model_kind == "ar":
                st.markdown(f"- **自變數**: {target} 的前 {ar_p} 期 ($y_{{t-1}},\\dots,y_{{t-{ar_p}}}$)")
                st.markdown(f"- **應變數** ($y_t$): {target}")
                st.markdown(f"- **模型假設**: $y_t = b + \\phi_1y_{{t-1}} + \\cdots + \\phi_{{{ar_p}}}y_{{t-{ar_p}}} + \\epsilon_t$ (AR({ar_p}))")
            elif model_kind == "simple":
                st.markdown(f"- **自變數** ($x$): {feats[0]}")
                st.markdown(f"- **應變數** ($y$): {target}")
                st.markdown("- **模型假設**: $y = ax + b + \\epsilon,\\ \\epsilon \\sim N(0,\\sigma^2)$")
            else:
                st.markdown(f"- **自變數** ($x_1,\\dots,x_p$): {', '.join(feats)}")
                st.markdown(f"- **應變數** ($y$): {target}")
                st.markdown("- **模型假設**: $y = b + w_1x_1 + w_2x_2 + \\cdots + w_px_p + \\epsilon,\\ \\epsilon \\sim N(0,\\sigma^2)$")
            st.markdown("- **分析任務**: 以**公式解** $\\hat\\beta=(X^TX)^{-1}X^Ty$ 與**梯度下降法**估計係數並比較。")

# ---------------- 2. 數據理解 ----------------
with tabs[1]:
    st.header("📊 第二階段: 數據理解 (Data Understanding)")
    st.caption(f"資料來源: {source}")
    m = st.columns(4)
    m[0].metric("樣本數", f"{len(raw)} 筆")
    m[1].metric("欄位數", f"{raw.shape[1]}")
    m[2].metric("缺失值總數", int(raw.isna().sum().sum()))
    m[3].metric(f"{target} 標準差", f"{y_s.std():.2f}")
    st.subheader("📄 前 10 筆資料")
    st.dataframe(raw.head(10), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📐 敘述統計")
        st.dataframe(raw.describe(include="all").T, use_container_width=True)
    with c2:
        st.subheader("🧱 欄位型態與缺失值")
        st.dataframe(pd.DataFrame({"型態": raw.dtypes.astype(str), "缺失值": raw.isna().sum()}),
                     use_container_width=True)
    num_df = clean[feats + [target]].select_dtypes("number")
    if model_kind == "ar":
        st.subheader(f"⏱️ {target} 時間序列圖與自相關函數 (ACF)")
        cA, cB = st.columns(2)
        cA.plotly_chart(px.line(series_all, labels={"index": "時間索引 t", "value": target}), use_container_width=True)
        nl = min(20, len(series_all) // 2)
        ac = acf(series_all.values, nl)
        cB.plotly_chart(px.bar(x=list(range(nl + 1)), y=ac, labels={"x": "落後期數 k", "y": "自相關"}),
                        use_container_width=True)
        st.caption("提醒: 內建的 50 Startups / Boston Housing 並非真正的時間序列, AR 僅依資料列順序示範; "
                   "自訂時間序列資料 (如股價、氣溫) 才最能展現 AR 效果。")
    c1, c2 = st.columns(2)
    with c1:
        if num_df.shape[1] >= 2:
            st.subheader("🔥 相關係數熱圖")
            fig = px.imshow(num_df.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader(f"📈 {target} 分佈")
        st.plotly_chart(px.histogram(y_s, nbins=30, labels={"value": target}), use_container_width=True)
    if num_df.shape[1] >= 2:
        st.subheader("🔍 特徵與目標變數散佈圖")
        pick = st.selectbox("選擇特徵", [c for c in num_df.columns if c != target], key="scatter_pick")
        st.plotly_chart(px.scatter(clean, x=pick, y=target, trendline="ols",
                                   color=cat_cols[0] if cat_cols else None), use_container_width=True)

# ---------------- 3. 數據準備 ----------------
with tabs[2]:
    st.header("🧹 第三階段: 數據準備 (Data Preparation)")
    st.write("包含: **缺失值處理、類別變數 One-Hot 編碼、訓練/測試切分、特徵縮放說明、離群值標記**。")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("🧴 前處理摘要")
            st.markdown(f"- 數值欄位缺失值以**中位數**補齊、類別欄位以**眾數**補齊")
            st.markdown(f"- 類別欄位 ({'One-Hot, 移除一個基準類別' if onehot_drop else '完整 One-Hot, 每個類別一欄'}): {', '.join(cat_cols) if cat_cols else '無'}")
            if model_kind == "multiple" and any(c not in feats for c in cols if c != target):
                st.markdown(f"- 未使用的原始欄位: {', '.join(c for c in cols if c != target and c not in feats)} (可在側邊欄「特徵變數」加回)")
            st.markdown(f"- 編碼後候選特徵數: **{X_full_df.shape[1]}** → {', '.join(X_full_df.columns)}")
            st.markdown(f"- 特徵篩選後 (見下方「特徵篩選過程」): **{len(names)}** 個 → {', '.join(names)}")
            if model_kind == "ar":
                st.markdown(f"- 以前 **{ar_p}** 期的 y 建立落後特徵 (損失前 {ar_p} 筆), 共 **{len(X_all)}** 筆可用樣本")
                st.markdown(f"- 依時間順序切分: 訓練集 **{len(X_tr)}** 筆 / 測試集 **{len(X_te)}** 筆 (不隨機打亂)")
            else:
                st.markdown(f"- 訓練集 **{len(X_tr)}** 筆 / 測試集 **{len(X_te)}** 筆 (seed={seed})")
            st.markdown(f"- 梯度下降特徵縮放: **{'啟用 (標準化 X 與 y)' if use_scaling else '未啟用'}**")
    with c2:
        st.subheader("🧾 編碼後的 X (前 10 筆)")
        st.dataframe(X_df.head(10), use_container_width=True)

    if cat_cols:
        st.markdown("**🔢 One-Hot 編碼對照** — " + ("每個類別變數展開成「類別數 − 1」個 0/1 欄位, 被移除的基準類別以全 0 表示 (可在上方更換基準)"
                    if onehot_drop else "每個類別變數展開成「類別數」個 0/1 欄位 (如 100 / 010 / 001)"))
        mcols = st.columns(len(cat_cols))
        for mc, c in zip(mcols, cat_cols):
            levels = sorted(clean[c].astype(str).unique())
            eye = np.eye(len(levels), dtype=int)
            keep_idx = [i for i, l in enumerate(levels) if not (onehot_drop and l == onehot_base.get(c, levels[0]))]
            use_cols = [levels[i] for i in keep_idx]
            mat = eye[:, keep_idx]
            oh = pd.DataFrame(mat, columns=[f"{c}_{l}" for l in use_cols])
            oh.insert(0, c, levels)
            oh["編碼"] = ["".join(map(str, r)) for r in mat]
            with mc:
                st.dataframe(oh, use_container_width=True, hide_index=True)
        if onehot_drop:
            st.caption("已移除基準類別 (全 0), 避免虛擬變數陷阱: 例如 State 有 3 個州只需 2 欄, 基準州以 00 表示; 換基準不會改變預測, 只改變係數的解讀。")
        else:
            st.caption("⚠️ 完整 One-Hot 的各欄相加恆為 1, 與截距項完全共線 (虛擬變數陷阱): 預測結果不受影響 "
                       "(sklearn 與公式解皆取最小範數解), 但這幾欄的 VIF 會是 inf、係數的顯著性檢定僅供參考; "
                       "多出來的那一欄也不帶新資訊 (特徵數 k 最後一步 RMSE 不會再變)。")

    # ================= 特徵篩選過程 =================
    st.divider()
    st.subheader("🔎 特徵篩選過程 (Feature Selection)")
    if model_kind == "simple":
        st.write("單純線性回歸只用一個特徵, 這裡比較所有數值型候選特徵與目標變數的關聯強度, 選出最佳者 (側邊欄可手動更換)。")
        cand_X = prepare_xy(raw, target, num_cols)[0]
        diag = feature_diagnostics(cand_X.iloc[idx_tr], y_tr)
    elif model_kind == "ar":
        st.write("自迴歸的「特徵」是各期落後項, 這裡檢視每一個落後期與 y(t) 的關聯, 協助判斷階數 p 是否合理。")
        diag = feature_diagnostics(X_full_df.iloc[idx_tr], y_tr)
    else:
        st.write("為避免資料洩漏, **篩選只使用訓練集**。流程: ① 計算每個特徵的診斷指標 → ② 依所選方法篩選 → ③ 比較篩選前後模型表現。")
        diag = feature_diagnostics(X_full_df.iloc[idx_tr], y_tr)

    st.markdown("**步驟 ① 各特徵診斷指標** — 依三大類篩選方法分類 (Filter / Wrapper / Embedded)")
    diag = diag.assign(是否選用=diag["特徵"].isin(selected).map({True: "✅", False: "—"}))
    fmt = {"|r| 與 y": "{:.3f}", "F 值": "{:.2f}", "p-value": "{:.4f}", "互資訊": "{:.3f}", "VIF": "{:.2f}",
           "Lasso 係數": "{:.3f}", "RF 重要度": "{:.3f}"}
    t_f, t_w, t_e = st.tabs(["Filter 過濾法", "Wrapper 包裝法", "Embedded 嵌入法"])
    for tb, cat, cs in [(t_f, "Filter 過濾法", ["|r| 與 y", "F 值", "p-value", "互資訊", "VIF"]),
                        (t_w, "Wrapper 包裝法", ["RFE 排名"]),
                        (t_e, "Embedded 嵌入法", ["Lasso 係數", "RF 重要度"])]:
        with tb:
            st.caption(FS_DESC[cat])
            sort_c = {"Filter 過濾法": "F 值", "Wrapper 包裝法": "RFE 排名", "Embedded 嵌入法": "RF 重要度"}[cat]
            sub = diag[["特徵"] + cs + ["是否選用"]].sort_values(sort_c, ascending=(cat == "Wrapper 包裝法"))
            st.dataframe(sub.style.format({k: v for k, v in fmt.items() if k in cs}), use_container_width=True)
    fig = px.bar(diag.sort_values("|r| 與 y"), x="|r| 與 y", y="特徵", orientation="h", color="是否選用",
                 color_discrete_map={"✅": "#4f46e5", "—": "#c7c7d1"}, title="各特徵與目標變數的絕對相關係數 |r|")
    st.plotly_chart(fig, use_container_width=True)

    if model_kind == "multiple" and X_full_df.shape[1] >= 2:
        thr_txt = ("不篩選" if fs_method == "none" else f"|r| ≥ {c_thr}" if fs_method == "corr"
                   else f"p-value 門檻 {p_thr}" if fs_method == "backward"
                   else f"k = {k_keep}" if fs_method in K_BASED else "由模型/CV 自動決定")
        st.markdown(f"**步驟 ② 篩選結果** — 類別: `{FS_CAT_OF.get(fs_method, '')}` / 方法: `{fs_name}` ({thr_txt})")
        dropped = [c for c in X_full_df.columns if c not in selected]
        st.success(f"保留 {len(selected)} 個特徵: {', '.join(selected)}")
        if dropped:
            st.info(f"剔除 {len(dropped)} 個特徵: {', '.join(dropped)}")
        if fs_log:
            st.markdown("向後逐步消除紀錄 (每一輪剔除 p-value 最大且超過門檻的特徵; p-value = 1.0 表示與截距共線的 One-Hot 欄位, 視為基準類別先移除):")
            st.dataframe(pd.DataFrame([{"輪次": i + 1, "剔除特徵": n, "當輪 p-value": pv}
                                       for i, (n, pv) in enumerate(fs_log)]), use_container_width=True)
        st.markdown("**步驟 ③ 各類方法保留的特徵一覽**")
        if board_reg is not None:
            st.dataframe(board_reg[["分類", "模型", "特徵數", "保留特徵"]], use_container_width=True, hide_index=True)
            st.caption("各方法的 CV / 測試表現與最佳模型, 請見「4. 建立模型」分頁的「🏆 最佳模型選擇」; "
                       "特徵數量與 RMSE / R² 的關係圖在「5. 模型評估」分頁最後。")
        else:
            st.caption("側邊欄勾選「執行最佳模型搜尋」即可列出所有方法保留的特徵與績效。")
    elif model_kind == "simple":
        st.success(f"選用特徵: {feats[0]}")

    b_all, w_all = ols_fit(X_all, y_all)
    resid_all = y_all - pred(X_all, b_all, w_all)
    out_idx = np.argsort(-np.abs(resid_all))[:n_outliers]
    st.subheader(f"🚨 {n_outliers} 個離群值 (依 OLS「絕對殘差最大」)")
    out_df = pd.DataFrame({"index": X_df.index[out_idx], target: y_all[out_idx],
                           "預測值": pred(X_all[out_idx], b_all, w_all),
                           "殘差": resid_all[out_idx], "abs_residual": np.abs(resid_all[out_idx])})
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=pred(X_all, b_all, w_all), y=y_all, mode="markers", name="正常點",
                        marker=dict(color="lightgray"))
        fig.add_scatter(x=out_df["預測值"], y=out_df[target], mode="markers", name=f"{n_outliers} 個離群值",
                        marker=dict(color="crimson", symbol="x", size=11))
        lo, hi = float(min(y_all.min(), pred(X_all, b_all, w_all).min())), float(max(y_all.max(), pred(X_all, b_all, w_all).max()))
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = ŷ", line=dict(dash="dash", color="seagreen"))
        fig.update_layout(xaxis_title="預測值 ŷ", yaxis_title=f"實際值 {target}")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(out_df.round(3), use_container_width=True)

# ---------------- 4. 建立模型 ----------------
with tabs[3]:
    st.header("🛠️ 第四階段: 建立模型 (Modeling)")
    st.write(f"以 **sklearn `LinearRegression`** 作為主要模型估計「{MODEL_SHORT}」的參數, "
             "並與手刻的**公式解 (Normal Equation)**、**梯度下降法**比較係數。")
    if model_kind == "multiple":
        st.info(f"本模型使用的特徵 (經「{fs_name}」篩選): **{', '.join(names)}**")
    st.code(
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.model_selection import train_test_split, cross_val_score\n"
        "from sklearn.feature_selection import RFE, SelectKBest, f_regression, SequentialFeatureSelector\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        "# (特徵篩選只在訓練集上進行) → X_train = X_train[selected_features]\n"
        "model = LinearRegression().fit(X_train, y_train)\n"
        "y_pred = model.predict(X_test)\n"
        "print(model.intercept_, model.coef_)", language="python")
    if model_kind == "ar":
        st.latex(rf"y_t = b + \phi_1 y_{{t-1}} + \cdots + \phi_{{{ar_p}}} y_{{t-{ar_p}}} + \epsilon_t")
    if not gd_ok:
        st.error("⚠️ 梯度下降發散 (loss 變成無限大)。請降低學習率, 或開啟特徵縮放。")
    coef_cmp = pd.DataFrame({
        "變數": ["截距 (Intercept)"] + names,
        "sklearn": np.r_[b_sk, w_sk],
        "公式解 (OLS)": np.r_[b_ols, w_ols],
        "梯度下降 (GD)": np.r_[b_gd, w_gd] if gd_ok else np.nan,
    })
    coef_cmp["差距 |sklearn-GD|"] = (coef_cmp["sklearn"] - coef_cmp["梯度下降 (GD)"]).abs()
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("📋 係數比較")
        st.dataframe(coef_cmp.style.format({c: "{:.4f}" for c in coef_cmp.columns[1:]}), use_container_width=True)
        eq = " + ".join(f"{w:.3g}·{n}" for w, n in zip(w_ols, names))
        st.markdown(f"**OLS 方程式**: `{target} = {b_ols:.4g} + {eq}`")
    with c2:
        st.subheader("📊 係數長條圖 (不含截距)")
        bar = go.Figure()
        bar.add_bar(x=names, y=w_ols, name="OLS")
        if gd_ok:
            bar.add_bar(x=names, y=w_gd, name="GD")
        bar.update_layout(barmode="group", yaxis_title="係數")
        st.plotly_chart(bar, use_container_width=True)
    if model_kind == "simple":
        st.subheader(f"📐 單純線性回歸擬合線: {target} vs {feats[0]}")
        xs = X_all[:, 0]
        gx = np.linspace(xs.min(), xs.max(), 100)
        fig = go.Figure()
        fig.add_scatter(x=X_tr[:, 0], y=y_tr, mode="markers", name="訓練集", marker=dict(color="royalblue", opacity=.6))
        fig.add_scatter(x=X_te[:, 0], y=y_te, mode="markers", name="測試集", marker=dict(color="orange", opacity=.8))
        fig.add_scatter(x=gx, y=b_ols + w_ols[0] * gx, mode="lines", name=f"OLS: y={b_ols:.3g}+{w_ols[0]:.3g}x")
        if gd_ok:
            fig.add_scatter(x=gx, y=b_gd + w_gd[0] * gx, mode="lines", name=f"GD: y={b_gd:.3g}+{w_gd[0]:.3g}x",
                            line=dict(dash="dash"))
        fig.update_layout(xaxis_title=feats[0], yaxis_title=target)
        st.plotly_chart(fig, use_container_width=True)
    if gd_ok:
        st.subheader("📉 梯度下降損失曲線 (MSE)")
        ylab = "MSE (標準化後)" if use_scaling else "MSE"
        st.plotly_chart(px.line(y=losses, labels={"x": "迭代次數", "y": ylab}, log_y=True), use_container_width=True)
    st.subheader("🧪 係數顯著性檢定 (OLS, 訓練集)")
    st.dataframe(coef_table(X_tr, y_tr, names, b_ols, w_ols).style.format(
        {"係數": "{:.4f}", "標準誤": "{:.4f}", "t 值": "{:.3f}", "p-value": "{:.4f}"}), use_container_width=True)

    # ================= 最佳模型選擇 =================
    st.divider()
    st.subheader("🏆 最佳模型選擇 (Model Selection)")
    board = {"multiple": board_reg, "simple": board_simple, "ar": board_ar}[model_kind]
    if board is None:
        st.info("側邊欄勾選「執行最佳模型搜尋」後, 這裡會列出所有候選模型的比較。" if model_kind == "multiple"
                else "資料筆數太少, 無法進行模型搜尋。")
    else:
        pool = board[board["k"] >= 1] if model_kind == "ar" else board
        best_r, sugg_r = pick_best(pool)
        cand_desc = {"multiple": "候選 = 『單純線性回歸』+『全部特徵』+ Filter / Wrapper / Embedded 各種篩選方法 (最終估計器皆為 sklearn LinearRegression)",
                     "simple": "候選 = 每一個數值特徵各建一個單純線性回歸",
                     "ar": "候選 = 樸素預測 y(t)=y(t-1) 與 AR(1) ~ AR(p_max)"}[model_kind]
        st.write(f"{cand_desc}。評分準則: **訓練集 5 折交叉驗證 RMSE (越低越好)**; "
                 + ("特徵篩選在每一折內重做, " if model_kind == "multiple" else "")
                 + "測試集只用於最後驗證, 不參與選模。")
        show = board.copy()
        show["排名"] = show["cv_rmse"].rank(method="min").astype(int)

        def _mark(r):
            tag = []
            if r["模型"] == best_r["模型"]:
                tag.append("🏆 最佳")
            if r["模型"] == sugg_r["模型"]:
                tag.append("⭐ 建議")
            cur = (r["k"] == ar_p and r["特徵數"] >= 1) if model_kind == "ar" else (set(r["support"]) == set(names))
            if cur:
                tag.append("✅ 目前採用")
            return " ".join(tag)

        show["標記"] = show.apply(_mark, axis=1)
        show = show.sort_values("cv_rmse")
        cols_show = ["排名", "標記", "分類", "模型", "特徵數", "保留特徵", "cv_rmse", "cv_std", "cv_r2", "test_rmse", "test_r2"]
        cols_show = [c for c in cols_show if not (c == "分類" and model_kind != "multiple")]
        tbl = show[cols_show].rename(columns={"cv_rmse": "CV RMSE", "cv_std": "CV 標準差", "cv_r2": "CV R²",
                                              "test_rmse": "測試 RMSE", "test_r2": "測試 R²"})
        st.dataframe(tbl.style.format({"CV RMSE": "{:.4f}", "CV 標準差": "{:.4f}", "CV R²": "{:.4f}",
                                       "測試 RMSE": "{:.4f}", "測試 R²": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        figb = px.bar(show.iloc[::-1], x="cv_rmse", y="模型", orientation="h", error_x="cv_std",
                      color="分類" if model_kind == "multiple" else None,
                      labels={"cv_rmse": "CV RMSE (越低越好)", "模型": ""}, title="各候選模型的 5 折 CV RMSE (誤差線 = 標準差)")
        figb.update_layout(height=max(320, 26 * len(show) + 120))
        st.plotly_chart(figb, use_container_width=True)
        st.success(f"🏆 **最佳模型 (CV RMSE 最低)**: {best_r['模型']} — 特徵 [{best_r['保留特徵']}]; "
                   f"CV RMSE = {best_r['cv_rmse']:.4f} ± {best_r['cv_std']:.4f}, 測試 RMSE = {best_r['test_rmse']:.4f}, "
                   f"測試 R² = {best_r['test_r2']:.4f}")
        st.info(f"⭐ **建議模型 (1-SE 規則)**: {sugg_r['模型']} — 特徵 [{sugg_r['保留特徵']}]; "
                f"CV RMSE = {sugg_r['cv_rmse']:.4f}。1-SE 規則: 在『CV RMSE 與最佳者相差不到 1 個標準誤』的模型中, "
                "選擇特徵最少的, 以降低過度擬合風險並提高可解釋性。")
        if model_kind == "multiple":
            st.caption("在側邊欄「🏆 最佳模型選擇 → 模型採用方式」選擇自動採用, 即可讓整個 CRISP-DM 流程改用該模型; "
                       "勾選「同時搜尋最佳特徵數 k」會把 k = 1, 2, 3, 5, 8 也納入候選。")
        elif model_kind == "simple":
            st.caption("在側邊欄「單一特徵選擇方式」選擇「🏆 自動」, 即可改用 CV RMSE 最低的單一特徵。")
        else:
            st.caption("在側邊欄「自迴歸階數 p 選擇方式」選擇自動, 即可改用最佳階數; 樸素預測僅作為比較基準。")

# ---------------- 5. 模型評估 ----------------
with tabs[4]:
    st.header("📈 第五階段: 模型評估 (Evaluation)")
    if model_kind == "multiple":
        st.caption(f"使用特徵 ({len(names)} 個, 篩選方法: {fs_name}): {', '.join(names)}")
    rows = []
    for nm, (b, w, ok) in {"sklearn LinearRegression": (b_sk, w_sk, True), "公式解 (OLS)": (b_ols, w_ols, True),
                           "梯度下降 (GD)": (b_gd, w_gd, gd_ok)}.items():
        if not ok:
            continue
        for sp, (Xs_, ys_) in {"訓練集": (X_tr, y_tr), "測試集": (X_te, y_te)}.items():
            rows.append({"模型": nm, "資料": sp, **metrics(ys_, pred(Xs_, b, w), X_all.shape[1])})
    st.dataframe(pd.DataFrame(rows).style.format({k: "{:.4f}" for k in ["R²", "Adj R²", "RMSE", "MAE"]}),
                 use_container_width=True)

    splitter = TimeSeriesSplit(5) if model_kind == "ar" else KFold(5, shuffle=True, random_state=int(seed))
    cv_scores = cross_val_score(LinearRegression(), X_all, y_all, cv=splitter, scoring="r2")
    cv_name = "5 折時間序列交叉驗證 (TimeSeriesSplit)" if model_kind == "ar" else "5-Fold 交叉驗證"
    st.info(f"{cv_name} R² (sklearn LinearRegression): 平均 **{np.mean(cv_scores):.4f}** ± {np.std(cv_scores):.4f}  "
            f"(各折: {', '.join(f'{s:.3f}' for s in cv_scores)})")

    if model_kind == "ar":
        st.subheader("⏱️ 時間序列: 實際值 vs 預測值 (灰線右側為測試集)")
        t_idx = np.asarray(X_df.index)
        fig = go.Figure()
        fig.add_scatter(x=t_idx, y=y_all, mode="lines", name="實際值", line=dict(color="gray"))
        fig.add_scatter(x=t_idx, y=pred(X_all, b_ols, w_ols), mode="lines", name="OLS 預測")
        if gd_ok:
            fig.add_scatter(x=t_idx, y=pred(X_all, b_gd, w_gd), mode="lines", name="GD 預測", line=dict(dash="dash"))
        fig.add_vline(x=t_idx[len(X_tr)], line_dash="dot", line_color="red")
        fig.update_layout(xaxis_title="時間索引 t", yaxis_title=target)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎯 實際值 vs 預測值 (測試集)")
        fig = go.Figure()
        fig.add_scatter(x=y_te, y=pred(X_te, b_ols, w_ols), mode="markers", name="OLS")
        if gd_ok:
            fig.add_scatter(x=y_te, y=pred(X_te, b_gd, w_gd), mode="markers", name="GD", marker=dict(symbol="diamond-open"))
        fig.add_scatter(x=[y_te.min(), y_te.max()], y=[y_te.min(), y_te.max()], mode="lines",
                        name="理想線", line=dict(dash="dash", color="gray"))
        fig.update_layout(xaxis_title="實際值", yaxis_title="預測值")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("🧭 殘差圖 (測試集, OLS)")
        yh = pred(X_te, b_ols, w_ols)
        fig = px.scatter(x=yh, y=y_te - yh, labels={"x": "預測值", "y": "殘差"})
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader(f"🩺 {MODEL_SHORT}專屬診斷")
    p_feat = X_all.shape[1]
    resid_tr = y_tr - pred(X_tr, b_ols, w_ols)
    resid_te = y_te - pred(X_te, b_ols, w_ols)

    if model_kind == "simple":
        xv, yv = X_all[:, 0], y_all
        r = float(np.corrcoef(xv, yv)[0, 1])
        ct = coef_table(X_tr, y_tr, names, b_ols, w_ols)
        slope, se_s, p_s = float(ct.iloc[1]["係數"]), float(ct.iloc[1]["標準誤"]), float(ct.iloc[1]["p-value"])
        dof = len(X_tr) - 2
        tcrit = float(stats.t.ppf(0.975, dof))
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Pearson 相關係數 r", f"{r:.4f}")
        d2.metric("r² (全體資料)", f"{r ** 2:.4f}")
        d3.metric("斜率 p-value", f"{p_s:.4g}")
        d4.metric("斜率 95% 信賴區間", f"[{slope - tcrit * se_s:.4g}, {slope + tcrit * se_s:.4g}]")
        direction = "正" if slope > 0 else "負"
        sig = "顯著" if p_s < 0.05 else "不顯著"
        st.info(f"{feats[0]} 與 {target} 呈**{direction}相關**; {feats[0]} 每增加 1 單位, {target} 平均變化 **{slope:.4g}**, "
                f"在 5% 水準下**{sig}**。")
        st.plotly_chart(px.scatter(x=X_te[:, 0], y=resid_te, labels={"x": feats[0], "y": "殘差"},
                                   title=f"殘差 vs {feats[0]} (測試集)").add_hline(y=0, line_dash="dash", line_color="red"),
                        use_container_width=True)
        st.caption("單一特徵不會有多重共線性, 因此不顯示 VIF。")

    elif model_kind == "multiple":
        n_t = len(X_tr)
        r2_tr = metrics(y_tr, pred(X_tr, b_ols, w_ols), p_feat)["R²"]
        dof2 = n_t - p_feat - 1
        F = (r2_tr / p_feat) / ((1 - r2_tr) / dof2) if dof2 > 0 and r2_tr < 1 else np.nan
        pF = float(1 - stats.f.cdf(F, p_feat, dof2)) if np.isfinite(F) else np.nan
        d1, d2, d3 = st.columns(3)
        d1.metric("整體 F 統計量 (訓練集)", f"{F:.2f}")
        d2.metric("F 檢定 p-value", f"{pF:.4g}")
        d3.metric("顯著特徵數 (p<0.05)", int((coef_table(X_tr, y_tr, names, b_ols, w_ols).iloc[1:]["p-value"] < 0.05).sum()))
        st.caption("F 檢定: 檢驗「所有係數同時為 0」的假設; Adj R² 會懲罰多餘特徵, 可與 R² 對照判斷是否過度擬合。")
        st.subheader("🧮 多重共線性檢查 (VIF)")
        vif = vif_table(X_df)
        st.dataframe(vif.style.format({"VIF": "{:.2f}"}), use_container_width=True)
        if (vif["VIF"] > 10).any():
            st.warning("部分特徵 VIF > 10, 存在明顯多重共線性, 係數解讀需謹慎。")
        if cat_cols and not onehot_drop:
            st.caption("VIF = inf 來自完整 One-Hot 的虛擬變數陷阱 (同一類別變數的各欄相加恆為 1); "
                       "改選「移除一個基準類別」即可消除。")

    else:  # ar
        dw = float(np.sum(np.diff(resid_tr) ** 2) / np.sum(resid_tr ** 2))
        # 穩定性: 特徵多項式 z^p - phi1 z^(p-1) - ... - phi_p 的根需落在單位圓內
        roots = np.roots(np.r_[1.0, -w_ols])
        stable = bool(np.all(np.abs(roots) < 1))
        naive_pred = X_te[:, 0]  # 樸素預測: y(t) = y(t-1)
        m_ar = metrics(y_te, pred(X_te, b_ols, w_ols), p_feat)
        m_nv = metrics(y_te, naive_pred, 0)
        gain = (1 - m_ar["RMSE"] / m_nv["RMSE"]) * 100 if m_nv["RMSE"] else np.nan
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Durbin-Watson (殘差)", f"{dw:.3f}", help="約 2 表示殘差無自相關; <1.5 或 >2.5 需留意")
        d2.metric("AR 穩定性", "穩定 ✅" if stable else "不穩定 ⚠️", help="特徵根皆在單位圓內才穩定")
        d3.metric("AR 測試 RMSE", f"{m_ar['RMSE']:.4f}")
        d4.metric("樸素預測 RMSE (y(t)=y(t-1))", f"{m_nv['RMSE']:.4f}", f"{gain:+.1f}% 改善", delta_color="normal")
        if gain > 0:
            st.success(f"AR({ar_p}) 的測試 RMSE 比「直接用前一期當預測」低 {gain:.1f}%, 表示模型有額外預測價值。")
        else:
            st.warning("AR 模型未優於樸素預測 (y(t)=y(t-1)), 可嘗試調整階數 p, 或該序列本身接近隨機漫步。")
        nl = min(20, len(resid_tr) // 2)
        ac_r = acf(resid_tr, nl)
        band = 1.96 / np.sqrt(len(resid_tr))
        fig = px.bar(x=list(range(1, nl + 1)), y=ac_r[1:], labels={"x": "落後期數 k", "y": "殘差自相關"},
                     title="殘差 ACF (虛線為 95% 信賴帶, 超出代表模型仍有未解釋的自相關)")
        fig.add_hline(y=band, line_dash="dash", line_color="red")
        fig.add_hline(y=-band, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("AR 以前幾期的 y 作特徵, 落後項之間本來就相關, 因此不使用 VIF。")

    # ================= 特徵數量 vs 模型表現 (最後總結) =================
    st.divider()
    unit_cn = "AR 階數 p" if model_kind == "ar" else "特徵數量"
    st.subheader(f"📉 {unit_cn} vs 模型表現 (RMSE / R²)")
    sweep, xcol, xlabel, sweep_note = None, "特徵數", "Number of Features", ""
    if model_kind == "multiple" and X_full_df.shape[1] >= 2:
        # 與上方「特徵篩選方法」一致: 能指定 k 的方法直接沿用; 其他方法改用重要性 / 係數大小排序後取前 k 個
        if fs_method in K_BASED:
            sw_method, sw_name = fs_method, FS_NAME_OF[fs_method]
            how = "依該方法挑選 k 個特徵"
        else:
            rank_kind = {"lasso": "lasso", "enet": "enet", "rf": "rf", "corr": "corr"}.get(fs_method, "rfe")
            sw_method = "rank:" + rank_kind
            sw_name = FS_NAME_OF.get(fs_method, "RFE 遞迴特徵消除")
            how = {"lasso": "依 Lasso 標準化係數的絕對值大小排序後取前 k 個",
                   "enet": "依 ElasticNet 標準化係數的絕對值大小排序後取前 k 個",
                   "rf": "依隨機森林特徵重要度排序後取前 k 個",
                   "corr": "依 |r| 相關係數排序後取前 k 個"}.get(
                fs_method, "此方法不指定 k, 改以 RFE (標準化係數遞迴消除) 的順序取前 k 個")
        st.caption(f"📌 排序方法沿用上方設定：**{sw_name}** — {how}。")
        if sw_method in ("sfs", "sfs_bwd") and X_full_df.shape[1] > 8:
            st.caption("⚠️ SFS 在特徵很多時需要較長時間, 首次計算請稍候 (之後有快取)。")
        sweep = sweep_feature_count(X_full_df.iloc[idx_tr], y_tr, X_full_df.iloc[idx_te], y_te, sw_method,
                                    int(seed), float(c_thr), float(p_thr))
        sweep_note = f"以「{sw_name}」{how} (k = 1 … {X_full_df.shape[1]}), 用訓練集建模、在測試集評估"
    elif model_kind == "ar" and board_ar is not None:
        ar_sw = board_ar[board_ar["k"] >= 1]
        sweep = pd.DataFrame({"特徵數": ar_sw["k"].values, "保留特徵": ar_sw["保留特徵"].map(lambda t: f"[{t}]").values,
                              "RMSE": ar_sw["test_rmse"].values, "R²": ar_sw["test_r2"].values,
                              "CV RMSE (訓練集)": ar_sw["cv_rmse"].values, "CV 標準差": ar_sw["cv_std"].values})
        xcol, xlabel = "特徵數", "AR Order p"
        sweep_note = "AR(p) 的階數由 1 增加到上限, 依時間順序切分訓練/測試集"
    elif model_kind == "simple" and board_simple is not None:
        st.write("單純線性回歸只有 1 個特徵, 因此改比較「選哪一個特徵」:")
        bs = board_simple.sort_values("cv_rmse")
        figs = make_subplots(rows=1, cols=2, subplot_titles=("RMSE by Feature", "R-squared by Feature"))
        figs.add_trace(go.Bar(x=bs["保留特徵"], y=bs["test_rmse"], name="測試集 RMSE", marker_color="#3b78b5"), 1, 1)
        figs.add_trace(go.Bar(x=bs["保留特徵"], y=bs["cv_rmse"], name="CV RMSE (訓練集)", marker_color="#e08a1e"), 1, 1)
        figs.add_trace(go.Bar(x=bs["保留特徵"], y=bs["test_r2"], name="測試集 R²", marker_color="#3b78b5",
                              showlegend=False), 1, 2)
        figs.update_yaxes(title_text="RMSE", row=1, col=1)
        figs.update_yaxes(title_text="R-squared", row=1, col=2)
        figs.update_layout(height=420, barmode="group", legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(figs, use_container_width=True)
        tb = bs[["保留特徵", "test_rmse", "test_r2", "cv_rmse"]].rename(columns={
            "保留特徵": "Selected Feature", "test_rmse": "RMSE", "test_r2": "R-squared", "cv_rmse": "CV RMSE"})
        st.dataframe(tb.style.format({"RMSE": "{:.4f}", "R-squared": "{:.4f}", "CV RMSE": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        top = bs.iloc[0]
        st.markdown("#### 📖 圖表解讀")
        st.markdown(f"- **左圖 RMSE**: 預測誤差 (與 y 同單位), **越低越好**; **右圖 R²**: 模型解釋 y 變異的比例, **越高越好**。\n"
                    f"- 單獨使用 **{top['保留特徵']}** 的 CV RMSE 最低 ({top['cv_rmse']:.4f}), 測試 RMSE = {top['test_rmse']:.4f}, "
                    f"R² = {top['test_r2']:.4f}, 是單純線性回歸的最佳特徵。\n"
                    f"- 其他特徵與 y 的關聯較弱, 誤差較大、R² 較低; 若要進一步提升, 可改用多元線性回歸加入更多特徵。")

    if sweep is not None:
        st.caption(sweep_note)
        st.plotly_chart(count_chart(sweep, xcol, xlabel), use_container_width=True)
        tbl2 = sweep.rename(columns={"特徵數": "Number of Features" if model_kind != "ar" else "AR Order p",
                                     "保留特徵": "Selected Features", "R²": "R-squared"})
        st.dataframe(tbl2.style.format({"RMSE": "{:.6f}", "R-squared": "{:.6f}", "CV RMSE (訓練集)": "{:.4f}",
                                        "CV 標準差": "{:.4f}"}), use_container_width=True, hide_index=True)

        # ---------- 自動產生的解釋說明 ----------
        unit = " 個特徵" if model_kind != "ar" else " 階"
        pmax_x = int(sweep[xcol].max())
        r_rmse, r_r2 = sweep.loc[sweep["RMSE"].idxmin()], sweep.loc[sweep["R²"].idxmax()]
        r_cv = sweep.loc[sweep["CV RMSE (訓練集)"].idxmin()]
        ok = sweep[sweep["CV RMSE (訓練集)"] <= r_cv["CV RMSE (訓練集)"] + r_cv["CV 標準差"] / np.sqrt(5)]
        r_se = ok.sort_values(xcol).iloc[0]
        r_full = sweep.loc[sweep[xcol].idxmax()]
        rise = (r_full["RMSE"] / r_rmse["RMSE"] - 1) * 100
        drop = r_r2["R²"] - r_full["R²"]
        st.markdown("#### 📖 圖表解讀")
        lines = [
            "**左圖 (RMSE by Number of Features)**: RMSE 是預測誤差 (與 y 同單位), **越低越好**; "
            "**右圖 (R-squared)**: 模型解釋 y 變異的比例, **越高越好**。兩張圖的「谷底 / 峰頂」若落在同一個位置, 就是最佳特徵數。",
            f"**測試集 RMSE 最低** 出現在 **{int(r_rmse[xcol])}{unit}** {r_rmse['保留特徵']}: RMSE = {r_rmse['RMSE']:.4f}, "
            f"R² = {r_rmse['R²']:.4f}。",
            f"**測試集 R² 最高** 出現在 **{int(r_r2[xcol])}{unit}**: R² = {r_r2['R²']:.4f}。",
            f"**交叉驗證 (CV) RMSE 最低** 在 **{int(r_cv[xcol])}{unit}**; 依 1-SE 規則 (誤差與最佳者相差不到 1 個標準誤時選最精簡) "
            f"建議採用 **{int(r_se[xcol])}{unit}** {r_se['保留特徵']}。",
        ]
        if pmax_x > 1 and rise >= 3:
            lines.append(f"使用全部 {pmax_x}{unit}時 RMSE = {r_full['RMSE']:.4f}, 比最低值高出 **{rise:.1f}%**, "
                         f"R² 也比最高值低 {drop:.4f}。表示**後面加入的特徵幾乎沒有新資訊, 反而帶來雜訊與過度擬合 (overfitting)**, "
                         "所以曲線呈現「先下降、再上升」(R² 則相反) 的形狀。")
        elif pmax_x > 1:
            lines.append(f"使用全部 {pmax_x}{unit}時 RMSE = {r_full['RMSE']:.4f}, 與最低值只差 **{max(rise, 0):.1f}%** "
                         f"(R² 差 {drop:.4f}), 沒有明顯的過度擬合; 但曲線後段已趨於平坦, 代表**多加的特徵貢獻有限**。")
        if int(r_rmse[xcol]) != int(r_cv[xcol]):
            lines.append(f"測試集 ({int(r_rmse[xcol])}) 與交叉驗證 ({int(r_cv[xcol])}) 挑出的最佳數量不同, 表示這幾種數量的表現差異很小、"
                         "落在隨機雜訊範圍內; 此時依 1-SE 規則選較精簡者最穩健。")
        if model_kind == "multiple" and cat_cols and not onehot_drop and len(sweep) >= 2 \
                and abs(sweep["RMSE"].iloc[-1] - sweep["RMSE"].iloc[-2]) < 1e-6:
            lines.append(f"最後兩點 (k = {int(sweep[xcol].iloc[-2])} 與 {int(sweep[xcol].iloc[-1])}) 的 RMSE / R² **完全相同**: "
                         "完整 One-Hot 的最後一個州別欄可由其他欄推得 (三欄相加恆為 1), 不帶任何新資訊 — 這就是虛擬變數陷阱。")
        lines.append("**如何選擇**: 以曲線「手肘」(邊際效益遞減的轉折點) 為準, 在誤差差不多時優先選擇**較少的特徵** — "
                     "模型更簡單、更容易解釋, 對新資料也更穩定。")
        lines.append(f"⚠️ 測試集只有 **{len(y_te)}** 筆, 曲線會有隨機波動; 請同時參考虛線的 **CV RMSE** (訓練集 5 折, 篩選在折內進行, 較穩定), "
                     "並可更換側邊欄的 Seed 檢查結論是否一致。")
        for ln in lines:
            st.markdown(f"- {ln}")
        st.success(f"✅ **結論**: 建議使用 **{int(r_se[xcol])}{unit}** — {r_se['保留特徵']} "
                   f"(CV RMSE = {r_se['CV RMSE (訓練集)']:.4f}, 測試 RMSE = {r_se['RMSE']:.4f}, 測試 R² = {r_se['R²']:.4f})。")

# ---------------- 6. 模型部署 ----------------
with tabs[5]:
    st.header("🚀 第六階段: 模型部署 (Deployment)")
    st.write("提供互動預測、模型參數匯出與 Streamlit 部署說明。")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔮 線上預測模擬")
        if model_kind == "ar":
            st.caption(f"輸入最近 {ar_p} 期的 {target} (預設為資料最後 {ar_p} 筆), 預測下一期。")
            last_vals = series_all.iloc[::-1].iloc[:ar_p].tolist()  # y(t-1), y(t-2), ...
            lag_in = [st.number_input(f"y(t-{k})", value=float(last_vals[k - 1]), format="%.4f", key=f"lag_{k}")
                      for k in range(1, ar_p + 1)]
            xr = np.array([lag_in])
            horizon = st.slider("多步預測期數 (遞迴預測)", 1, 30, 10)
            hist = list(lag_in[::-1])  # 由舊到新
            fc_ols, fc_gd = [], []
            hist_gd = list(hist)
            for _ in range(horizon):
                x_o = np.array(hist[::-1][:ar_p])
                fc_ols.append(float(b_ols + w_ols @ x_o)); hist.append(fc_ols[-1])
                if gd_ok:
                    x_g = np.array(hist_gd[::-1][:ar_p])
                    fc_gd.append(float(b_gd + w_gd @ x_g)); hist_gd.append(fc_gd[-1])
            fig = go.Figure()
            tail = series_all.iloc[-min(60, len(series_all)):]
            fig.add_scatter(x=tail.index, y=tail.values, mode="lines", name="歷史資料")
            fx = list(range(len(series_all), len(series_all) + horizon))
            fig.add_scatter(x=fx, y=fc_ols, mode="lines+markers", name="OLS 預測")
            if gd_ok:
                fig.add_scatter(x=fx, y=fc_gd, mode="lines+markers", name="GD 預測", line=dict(dash="dash"))
            fig.update_layout(xaxis_title="時間索引 t", yaxis_title=target)
            p_ols = fc_ols[0]
            p_gd_val = fc_gd[0] if gd_ok else None
        else:
            # 只需輸入「被篩選後仍有用到」的原始特徵 (類別特徵的 One-Hot 欄位名稱為 特徵_類別)
            used = [f for f in feats if f in names or any(n.startswith(f"{f}_") for n in names)]
            used_cat = [c for c in cat_cols if c in used]
            st.caption(f"依特徵篩選結果, 模型使用 {len(used)} 個原始特徵: {', '.join(used)}")
            inputs = {}
            for f in used:
                if f in used_cat:
                    inputs[f] = st.selectbox(f, sorted(clean[f].astype(str).unique()), key=f"in_{f}")
                else:
                    col = clean[f].astype(float)
                    inputs[f] = st.number_input(f, value=float(col.median()), format="%.4f", key=f"in_{f}")
            row = pd.DataFrame([inputs])
            full = pd.concat([clean[used].astype({c: str for c in used_cat}), row], ignore_index=True)
            xr_df = pd.get_dummies(full, columns=used_cat, drop_first=False, dtype=float)
            if onehot_drop:
                xr_df = xr_df.drop(columns=[f"{c}_{onehot_base.get(c)}" for c in used_cat
                                            if f"{c}_{onehot_base.get(c)}" in xr_df.columns])
            xr = xr_df.iloc[[-1]].reindex(columns=names, fill_value=0.0).values
            p_ols = float(sk_model.predict(xr)[0])  # sklearn 預測 (與公式解相同)
            p_gd_val = float(pred(xr, b_gd, w_gd)[0]) if gd_ok else None
            fig = None
        nxt = "下一期" if model_kind == "ar" else ""
        st.success(f"**sklearn / 公式解 (OLS) 預測{nxt} {target}: {p_ols:,.4f}**")
        if gd_ok:
            st.warning(f"**梯度下降 (GD) 預測{nxt} {target}: {p_gd_val:,.4f}**")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("📦 匯出模型參數與原始數據")
        st.download_button("💾 下載資料 (CSV)", raw.to_csv(index=False).encode("utf-8-sig"),
                           "dataset.csv", "text/csv", use_container_width=True)
        params = {"model": model_name, "target": target, "feature_selection": fs_name, "features": names,
                  "sklearn_linear_regression": {"intercept": b_sk, "weights": dict(zip(names, map(float, w_sk)))},
                  "ols": {"intercept": float(b_ols), "weights": dict(zip(names, map(float, w_ols)))}}
        if gd_ok:
            params["gd"] = {"intercept": float(b_gd), "weights": dict(zip(names, map(float, w_gd))),
                            "lr": lr, "iterations": int(iters), "feature_scaling": use_scaling}
        st.download_button("💾 下載模型參數 (JSON)", json.dumps(params, ensure_ascii=False, indent=2).encode("utf-8"),
                           "model_params.json", "application/json", use_container_width=True)
        with st.container(border=True):
            st.subheader("☁️ 上架至 Streamlit Community Cloud")
            st.markdown("1. 將 `main.py`、`requirements.txt` 與 `data/` 資料夾上傳至 GitHub 儲存庫。\n"
                        "2. 前往 [share.streamlit.io](https://share.streamlit.io) 並登入。\n"
                        "3. 點擊 **New app**, 選定儲存庫、分支與主檔案 `main.py`。\n"
                        "4. 點擊 **Deploy**, 幾分鐘後即可公開存取。")
