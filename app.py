# app.py — Bot prédictif ultra-complet (tout-en-un)
# Requirements minimal: flask requests numpy scikit-learn pandas matplotlib
# Recommended optional packages for full features:
# xgboost lightgbm tensorflow lifelines river shap ruptures mapie stable-baselines3

import time, threading, traceback, uuid, io, math, os
from collections import deque, defaultdict
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request, send_file, Response
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Scikit-learn basics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Optional libs detection (graceful fallback)
HAS_XGB = False
HAS_LGB = False
HAS_TF = False
HAS_LIFELINES = False
HAS_RIVER = False
HAS_SHAP = False
HAS_RUPTURES = False
HAS_MAPIE = False
HAS_SB3 = False
try:
    import xgboost as xgb; HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import lightgbm as lgb; HAS_LGB = True
except Exception:
    HAS_LGB = False
try:
    import tensorflow as tf; HAS_TF = True
except Exception:
    HAS_TF = False
try:
    import lifelines; HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False
try:
    import river; HAS_RIVER = True
except Exception:
    HAS_RIVER = False
try:
    import shap; HAS_SHAP = True
except Exception:
    HAS_SHAP = False
try:
    import ruptures; HAS_RUPTURES = True
except Exception:
    HAS_RUPTURES = False
try:
    from mapie.conformal_prediction import MapieRegressor; HAS_MAPIE = True
except Exception:
    HAS_MAPIE = False
try:
    import stable_baselines3 as sb3; HAS_SB3 = True
except Exception:
    HAS_SB3 = False

# ---------------- CONFIG ----------------
URL_HISTORY = "https://crash-gateway-grm-cr.100hp.app/history"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://1play.gamedev-tech.cc",
    "referer": "https://1play.gamedev-tech.cc/",
    "customer-id": "077dee8d-c923-4c02-9bee-757573662e69",
    "session-id": "1f56b906-1fe1-4cc8-add1-34222b858a7e",
    "user-agent": "Mozilla/5.0"
}

POLL_INTERVAL = 4
TRAIN_INTERVAL = 60
WINDOW_SIZE = 30
MAX_HORIZON_SECONDS = 15 * 60
BUFFER_MAX = 60000
TARGETS = [2.0, 3.0, 5.0, 10.0]
CONF_THRESH = {2: 0.50, 3: 0.55, 5: 0.60, 10: 0.65}
FAILURES_TO_COOLDOWN = 3
MIN_COOLDOWN_SECONDS = 60
MAX_COOLDOWN_SECONDS = 60 * 30
USE_CALIBRATION = True  # if sklearn calibrator available
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------- BUFFER ----------------
class ExperienceBuffer:
    def __init__(self, maxlen=BUFFER_MAX):
        self.rounds = deque(maxlen=maxlen)  # dict: {"id","crash","ts"}
        self.ids = set()
        self.lock = threading.Lock()

    def append_raw(self, r):
        rid = r.get("round_id") or r.get("id") or None
        cp = r.get("crash_point") if "crash_point" in r else r.get("crash") if "crash" in r else None
        ts = r.get("timestamp") or r.get("ts") or None
        if cp is None: return
        if rid is not None and rid in self.ids: return
        if ts is None:
            ts = int(time.time())
        entry = {"id": rid, "crash": float(cp), "ts": int(ts)}
        with self.lock:
            self.rounds.append(entry)
            if rid: self.ids.add(rid)

    def extend(self, arr):
        for r in arr:
            try:
                self.append_raw(r)
            except Exception:
                pass

    def get_series(self):
        with self.lock:
            crashes = [r["crash"] for r in self.rounds]
            timestamps = [r["ts"] for r in self.rounds]
        return crashes, timestamps

    def dump_df(self):
        crashes, ts = self.get_series()
        return pd.DataFrame({"crash": crashes, "ts": ts})

    def __len__(self):
        with self.lock:
            return len(self.rounds)

buffer = ExperienceBuffer()

# ---------------- HELPERS: mapping & features ----------------
def map_to_fixed(v):
    v = float(v)
    if v < 2.0: return 0.0
    if v < 2.5: return 2.0
    if v < 4.0: return 3.0
    if v < 7.5: return 5.0
    return 10.0

def seq_to_features(seq):
    arr = np.array(seq, dtype=float)
    mean = float(np.mean(arr)) if len(arr)>0 else 0.0
    std = float(np.std(arr)) if len(arr)>0 else 0.0
    last = float(arr[-1]) if len(arr)>0 else 0.0
    mn = float(np.min(arr)) if len(arr)>0 else 0.0
    mx = float(np.max(arr)) if len(arr)>0 else 0.0
    med = float(np.median(arr)) if len(arr)>0 else 0.0
    prop_low = float(np.sum(arr < 2.0)/len(arr)) if len(arr)>0 else 0.0
    pad = max(0, WINDOW_SIZE - len(arr))
    if pad>0:
        padded = np.concatenate([np.full(pad, mean), arr])
    else:
        padded = arr[-WINDOW_SIZE:]
    flat = padded.tolist()
    feats = flat + [mean, std, last, mn, mx, med, prop_low]
    return np.array(feats, dtype=float)

def compute_intervals(crashes, timestamps):
    by_target = {t: [] for t in TARGETS}
    for c,ts in zip(crashes, timestamps):
        m = map_to_fixed(c)
        if m in by_target: by_target[m].append(ts)
    out = {}
    for t, times in by_target.items():
        if len(times)>=2:
            diffs = [times[i]-times[i-1] for i in range(1,len(times))]
            out[int(t)] = {"avg": float(np.mean(diffs)), "last": float(diffs[-1]), "count": len(times), "last_ts": times[-1]}
        elif len(times)==1:
            out[int(t)] = {"avg": None, "last": None, "count":1, "last_ts": times[-1]}
        else:
            out[int(t)] = {"avg": None, "last": None, "count":0, "last_ts": None}
    return out

def seq_features_for_window_index(crashes, timestamps, idx_end):
    n = len(crashes)
    if n==0:
        base_seq = [0.0]*WINDOW_SIZE
        last_ts = int(time.time())
    else:
        ws = max(0, idx_end - WINDOW_SIZE + 1)
        seq = crashes[ws:idx_end+1]
        if len(seq) < WINDOW_SIZE:
            gm = float(np.mean(crashes))
            seq = [gm]*(WINDOW_SIZE - len(seq)) + seq
        base_seq = seq[-WINDOW_SIZE:]
        last_ts = timestamps[idx_end]
    feats = seq_to_features(base_seq)
    intervals = compute_intervals(crashes, timestamps)
    extra=[]
    for t in TARGETS:
        d = intervals[int(t)]
        extra.append(d["avg"] or 0.0)
        extra.append(d["last"] or 0.0)
        extra.append(d["count"] or 0)
        extra.append((time.time() - d["last_ts"]) if d["last_ts"] else 0.0)
    return np.concatenate([feats, np.array(extra, dtype=float)]), last_ts

def seq_features_for_last_window():
    crashes, timestamps = buffer.get_series()
    if len(crashes)==0:
        return np.zeros(WINDOW_SIZE+7+4*len(TARGETS)), int(time.time())
    return seq_features_for_window_index(crashes, timestamps, len(crashes)-1)

# ---------------- A: sliding stats / moving windows ----------------
def moving_stats(crashes, windows=(5,10,20,50,100)):
    res = {}
    n = len(crashes)
    arr = np.array(crashes) if n>0 else np.array([0.0])
    for w in windows:
        sub = arr[-w:] if n>=w else arr
        res[f"mv_{w}_mean"] = float(np.mean(sub))
        res[f"mv_{w}_std"] = float(np.std(sub))
        res[f"mv_{w}_min"] = float(np.min(sub))
        res[f"mv_{w}_max"] = float(np.max(sub))
    return res

# ---------------- B: pattern detection ----------------
def detect_repeated_sequence(crashes, k=6):
    if len(crashes) < k*2: return 0
    seq = crashes[-k:]
    count = 0
    for i in range(len(crashes) - k):
        if crashes[i:i+k] == seq:
            count += 1
    return count

# ---------------- DRIFT detection ----------------
class DriftDetector:
    def __init__(self, window=200, threshold=0.35):
        self.window = window
        self.threshold = threshold
        self.frozen = False
    def check_and_maybe_freeze(self, crashes):
        n = len(crashes)
        if n < self.window * 2: return False
        recent = np.array(crashes[-self.window:])
        past = np.array(crashes[-2*self.window:-self.window])
        mean_diff = abs(np.mean(recent) - np.mean(past)) / (abs(np.mean(past))+1e-9)
        std_diff = abs(np.std(recent) - np.std(past)) / (abs(np.std(past))+1e-9)
        score = (mean_diff + std_diff)/2.0
        if HAS_RUPTURES:
            try:
                import ruptures as rpt
                algo = rpt.Binseg(model="l2").fit(np.array(crashes))
                bkps = algo.predict(n_bkps=2)
                if any(b >= n - int(self.window/2) for b in bkps):
                    score = max(score, 0.5)
            except Exception:
                pass
        if score > self.threshold:
            self.frozen = True
            return True
        return False
    def reset(self):
        self.frozen = False

drift_detector = DriftDetector(window=200, threshold=0.35)

# ---------------- Survival models (time-to-event) ----------------
class SurvivalModels:
    def __init__(self):
        self.models = {}
        self.trained = {t: False for t in TARGETS}
    def train_from_buffer(self):
        crashes, timestamps = buffer.get_series()
        n = len(crashes)
        if n < WINDOW_SIZE + 60:
            return {"status":"not_enough"}
        rows=[]
        for i in range(0, n - WINDOW_SIZE - 1):
            feat, last_ts = seq_features_for_window_index(crashes, timestamps, i+WINDOW_SIZE-1)
            for j in range(i+WINDOW_SIZE, n):
                mapped = map_to_fixed(crashes[j])
                if mapped in TARGETS:
                    dt = timestamps[j] - last_ts
                    dt = min(dt, MAX_HORIZON_SECONDS)
                    rows.append({"feat": feat, "target": int(mapped), "dt": int(dt)})
                    break
        if len(rows)==0: return {"status":"no_labels"}
        df = pd.DataFrame(rows)
        for t in TARGETS:
            sub = df[df["target"]==int(t)]
            if len(sub) < 40:
                self.trained[int(t)] = False
                continue
            X = np.vstack(sub["feat"].values)
            durations = sub["dt"].values
            if HAS_LIFELINES:
                try:
                    from lifelines import CoxPHFitter
                    colnames = [f"f{i}" for i in range(X.shape[1])]
                    dfX = pd.DataFrame(X, columns=colnames)
                    dfX["T"] = durations
                    dfX["E"] = 1
                    cph = CoxPHFitter()
                    cph.fit(dfX, duration_col="T", event_col="E", show_progress=False)
                    self.models[int(t)] = ("cox", cph, colnames)
                    self.trained[int(t)] = True
                    continue
                except Exception:
                    pass
            try:
                r = RandomForestRegressor(n_estimators=80)
                r.fit(X, durations)
                self.models[int(t)] = ("rf", r)
                self.trained[int(t)] = True
            except Exception:
                self.trained[int(t)] = False
        return {"status":"trained_some", "trained": self.trained}
    def predict_time(self, feat):
        out={}
        for t in TARGETS:
            if not self.trained.get(int(t), False):
                out[int(t)] = None; continue
            m = self.models.get(int(t))
            if m is None:
                out[int(t)] = None; continue
            if m[0]=="cox" and HAS_LIFELINES:
                _, cph, cols = m
                try:
                    X = pd.DataFrame([feat], columns=cols)
                    surv = cph.predict_survival_function(X, times=np.arange(0, MAX_HORIZON_SECONDS, 10))
                    svals = surv.values.flatten()
                    idx = np.argmax(svals <= 0.5)
                    tpred = int(surv.index[idx]) if svals[idx] <= 0.5 else int(MAX_HORIZON_SECONDS)
                    out[int(t)] = max(120, min(MAX_HORIZON_SECONDS, tpred))
                except Exception:
                    out[int(t)] = None
            else:
                try:
                    r = m[1]
                    p = float(r.predict([feat])[0])
                    out[int(t)] = max(120, min(MAX_HORIZON_SECONDS, int(p)))
                except Exception:
                    out[int(t)] = None
        return out

survival_models = SurvivalModels()

# ---------------- Ensemble stacking + calibration ----------------
class EnsembleStack:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=150)
        self.gb = GradientBoostingClassifier(n_estimators=100)
        self.lr = LogisticRegression(max_iter=500)
        self.meta = LogisticRegression(max_iter=500)
        self.scaler = StandardScaler()
        self.calibrator = None
        self.trained = False
    def build_and_train(self):
        crashes, timestamps = buffer.get_series()
        n = len(crashes)
        X=[]; y=[]
        for i in range(0, n - WINDOW_SIZE - 1):
            feat, last_ts = seq_features_for_window_index(crashes, timestamps, i+WINDOW_SIZE-1)
            for j in range(i+WINDOW_SIZE, n):
                m = map_to_fixed(crashes[j])
                if m in TARGETS:
                    dt = timestamps[j] - last_ts
                    if dt <= MAX_HORIZON_SECONDS:
                        X.append(feat); y.append(int(m))
                    break
        if len(X) < 300:
            return {"status":"not_enough", "n":len(X)}
        X = np.vstack(X); y = np.array(y)
        Xs = self.scaler.fit_transform(X)
        self.rf.fit(Xs, y)
        self.gb.fit(Xs, y)
        self.lr.fit(Xs, y)
        metaX=[]
        for i in range(Xs.shape[0]):
            xi = Xs[i:i+1]
            p_rf = self.rf.predict_proba(xi)[0]
            p_gb = self.gb.predict_proba(xi)[0]
            p_lr = self.lr.predict_proba(xi)[0]
            metaX.append(np.concatenate([p_rf, p_gb, p_lr]))
        metaX = np.vstack(metaX)
        self.meta.fit(metaX, y)
        if USE_CALIBRATION:
            try:
                self.calibrator = CalibratedClassifierCV(self.meta, cv=3)
                self.calibrator.fit(metaX, y)
            except Exception:
                self.calibrator = None
        self.trained = True
        return {"status":"trained", "n_samples": len(y)}
    def predict_proba(self, feat):
        x = feat.reshape(1,-1)
        xs = self.scaler.transform(x) if hasattr(self.scaler, "mean_") else x
        if not self.trained:
            return {int(t): 1.0/len(TARGETS) for t in TARGETS}
        p_rf = self.rf.predict_proba(xs)[0]
        p_gb = self.gb.predict_proba(xs)[0]
        p_lr = self.lr.predict_proba(xs)[0]
        meta = np.concatenate([p_rf, p_gb, p_lr]).reshape(1,-1)
        if self.calibrator:
            probs = self.calibrator.predict_proba(meta)[0]
        else:
            probs = self.meta.predict_proba(meta)[0]
        classes = self.meta.classes_
        out = {int(t): 0.0 for t in TARGETS}
        for c,p in zip(classes, probs):
            out[int(c)] = float(p)
        s = sum(out.values())
        if s == 0:
            out = {int(t): 1.0/len(TARGETS) for t in TARGETS}
        else:
            for k in out:
                out[k] = out[k]/s
        return out

ensemble = EnsembleStack()

# ---------------- Online learner (river) fallback ----------------
class OnlineLearner:
    def __init__(self):
        self.enabled = HAS_RIVER
        if self.enabled:
            try:
                from river import tree, compose
                self.model = compose.Pipeline(tree.HoeffdingTreeClassifier())
            except Exception:
                self.enabled = False; self.model=None
        else:
            self.model = None
    def partial_fit(self, x_dict, y):
        if self.enabled and self.model:
            try:
                self.model.learn_one(x_dict, y)
            except Exception:
                pass
    def predict_proba_one(self, x_dict):
        if self.enabled and self.model:
            try:
                return self.model.predict_proba_one(x_dict)
            except Exception:
                return {}
        return {}

online = OnlineLearner()

# ---------------- Explainability (SHAP) ----------------
def compute_shap(feat):
    if not HAS_SHAP:
        return {"note":"shap not available"}
    try:
        explainer = shap.TreeExplainer(ensemble.rf)
        vals = explainer.shap_values(feat.reshape(1,-1))
        summary={}
        for i, cls in enumerate(ensemble.rf.classes_):
            arr = vals[i].flatten()
            idx = np.argsort(np.abs(arr))[::-1][:6]
            summary[int(cls)] = [{"feature_idx":int(k), "shap":float(arr[k])} for k in idx]
        return {"shap": summary}
    except Exception as e:
        return {"error": str(e)}

# ---------------- RL fallback: evolutionary strategy (if SB3 unavailable) ----------------
class EvolutionaryTrainer:
    """Simple evolutionary search over parameterized heuristic strategies.
       Each strategy = thresholds for selecting target and stake fraction.
       It's a practical fallback to 'train' a strategy on historical data.
    """
    def __init__(self, pop=20, gens=30):
        self.pop = pop
        self.gens = gens
        self.best = None
    def random_strategy(self):
        # returns dict: prob_thresholds for targets and stake fraction base
        return {
            "thr2": np.random.uniform(0.4, 0.9),
            "thr3": np.random.uniform(0.4, 0.9),
            "thr5": np.random.uniform(0.4, 0.9),
            "thr10": np.random.uniform(0.4, 0.9),
            "stake_frac": np.random.uniform(0.001, 0.05)
        }
    def evaluate(self, strat, df):
        # very simple backtest: when model prob >= thr -> bet stake_frac*bank on that target
        bank = 1000.0
        crashes = df["crash"].values
        n = len(crashes)
        for i in range(WINDOW_SIZE, n-1):
            # use ensemble probs approximated by frequency in recent window (naive)
            window = df["crash"].iloc[i-WINDOW_SIZE:i].values
            freqs = {2: np.mean(window >= 2), 3: np.mean(window >= 3), 5: np.mean(window >=5), 10: np.mean(window>=10)}
            chosen = None
            for t in [10,5,3,2]:
                if freqs[t] >= strat[f"thr{int(t)}"]:
                    chosen = t; break
            if chosen is None:
                continue
            stake = bank * strat["stake_frac"]
            # find next outcome in history
            found=False
            for j in range(i, n):
                if crashes[j] >= chosen:
                    payout = stake * (chosen - 1.0)
                    bank += payout
                    found=True; break
                if j==n-1 and not found:
                    bank -= stake
        return bank
    def train(self, df):
        # initialize
        pop = [self.random_strategy() for _ in range(self.pop)]
        scores = [self.evaluate(p, df) for p in pop]
        for g in range(self.gens):
            # select top half
            idx = np.argsort(scores)[-int(self.pop/2):]
            parents = [pop[i] for i in idx]
            children=[]
            for _ in range(int(self.pop/2)):
                a = parents[np.random.randint(len(parents))]
                b = parents[np.random.randint(len(parents))]
                # crossover + mutate
                child = {}
                for k in a:
                    child[k] = a[k] if np.random.rand()<0.5 else b[k]
                    if np.random.rand() < 0.2:
                        child[k] = child[k] * (1 + np.random.normal(0, 0.1))
                children.append(child)
            pop = parents + children
            scores = [self.evaluate(p, df) for p in pop]
        best_idx = int(np.argmax(scores))
        self.best = pop[best_idx]
        return {"best": self.best, "final_bank": scores[best_idx]}

evo_trainer = EvolutionaryTrainer(pop=20, gens=20)

# ---------------- Bankroll & sizing (Kelly) ----------------
def kelly_fraction(p, b):
    # p = prob of win, b = odds (payout per unit stake = target - 1)
    if b <= 0:
        return 0.0
    q = 1 - p
    k = (p * (b + 1) - 1) / b
    return max(0.0, min(1.0, k))

# ---------------- Pending predictions logger & monitor ----------------
class Predictor:
    def __init__(self):
        self.ensemble = ensemble
        self.survival = survival_models
        self.online = online
        self.drift = drift_detector
        self.failure_counts = {int(t): 0 for t in TARGETS}
        self.cooldowns = {int(t): 0 for t in TARGETS}
        self.pending = {}  # id->entry
        self.pending_lock = threading.Lock()
        self.recent = deque(maxlen=300)
        self.bandit_success = {int(t):1 for t in TARGETS}
        self.bandit_trials = {int(t):2 for t in TARGETS}
    def train_all(self):
        se = self.ensemble.build_and_train()
        ss = self.survival.train_from_buffer()
        return {"ensemble": se, "survival": ss}
    def score_bandit(self, t, model_prob):
        emp = self.bandit_success[t] / self.bandit_trials[t]
        return 0.6 * model_prob + 0.4 * emp
    def update_bandit(self, t, success):
        self.bandit_trials[t] += 1
        if success: self.bandit_success[t] += 1
    def dynamic_grace(self, target):
        crashes, timestamps = buffer.get_series()
        times = [ts for (c, ts) in zip(crashes, timestamps) if map_to_fixed(c) == target]
        if len(times) >= 2:
            diffs = [times[i] - times[i-1] for i in range(1, len(times))]
            avg = sum(diffs)/len(diffs)
            return int(min(MAX_HORIZON_SECONDS, max(30, avg/6)))
        default = {2:30, 3:60, 5:120, 10:180}
        return default.get(int(target), 60)
    def enter_cooldown(self, target):
        info = compute_intervals(*buffer.get_series()).get(int(target), {})
        avg = info.get("avg")
        if avg and avg>0:
            cd = int(min(MAX_COOLDOWN_SECONDS, max(MIN_COOLDOWN_SECONDS, avg/2)))
        else:
            cd = MIN_COOLDOWN_SECONDS * 3
        self.cooldowns[int(target)] = int(time.time()) + cd
        self.failure_counts[int(target)] = 0
    def in_cooldown(self, target):
        return time.time() < self.cooldowns.get(int(target), 0)
    def create_pending(self, target, pred_seconds, last_ts, recommended=True, grace=None):
        pid = str(uuid.uuid4())
        requested_at = int(time.time())
        pred_ts = last_ts + int(pred_seconds)
        if grace is None:
            grace = self.dynamic_grace(target)
        deadline = pred_ts + int(grace)
        entry = {"id": pid, "requested_at": requested_at, "target": int(target), "pred_ts": pred_ts, "deadline": deadline, "status": "pending", "recommended": recommended}
        with self.pending_lock:
            self.pending[pid] = entry
            self.recent.appendleft({"id": pid, "requested_at": requested_at, "requested_at_iso": datetime.fromtimestamp(requested_at).strftime("%Y-%m-%d %H:%M:%S"), "target": int(target), "pred_time_iso": datetime.fromtimestamp(pred_ts).strftime("%Y-%m-%d %H:%M:%S"), "status": "pending"})
        return entry
    def monitor_loop(self):
        while True:
            try:
                crashes, timestamps = buffer.get_series()
                now = int(time.time())
                with self.pending_lock:
                    keys = list(self.pending.keys())
                for k in keys:
                    with self.pending_lock:
                        e = self.pending.get(k)
                        if not e or e["status"] != "pending": continue
                        found=False
                        observed_ts=None
                        for c, ts in zip(crashes, timestamps):
                            if ts < e["requested_at"]: continue
                            if ts > e["deadline"]: continue
                            if map_to_fixed(c) == float(e["target"]):
                                found=True; observed_ts=ts; break
                        if found:
                            e["status"] = "success"; e["observed_at"] = observed_ts
                            self.update_bandit(int(e["target"]), True)
                            self.failure_counts[int(e["target"])] = 0
                        else:
                            if now > e["deadline"]:
                                e["status"] = "failure"
                                self.update_bandit(int(e["target"]), False)
                                self.failure_counts[int(e["target"])] = self.failure_counts.get(int(e["target"]),0) + 1
                                if self.failure_counts[int(e["target"])] >= FAILURES_TO_COOLDOWN:
                                    self.enter_cooldown(int(e["target"]))
                        self.pending[k] = e
                        # update recent
                        for rp in self.recent:
                            if rp["id"] == k:
                                rp["status"] = e["status"]; break
            except Exception:
                traceback.print_exc()
            time.sleep(2)
    def predict(self):
        crashes, timestamps = buffer.get_series()
        if len(crashes) < WINDOW_SIZE:
            return {"error": f"Pas assez de données ({len(crashes)} < {WINDOW_SIZE})"}
        if self.drift.check_and_maybe_freeze(crashes):
            return {"error":"Drift detected — frozen. Retrain required."}
        feat, last_ts = seq_features_for_last_window()
        repeated = detect_repeated_sequence(crashes, k=6)
        probs = self.ensemble.predict_proba(feat)
        surv = self.survival.predict_time(feat)
        results=[]
        for t in TARGETS:
            p = probs.get(int(t), 0.0)
            sec = surv.get(int(t), None)
            if sec is None:
                times_t = [ts for (c,ts) in zip(crashes, timestamps) if map_to_fixed(c) == t]
                if len(times_t)>=2:
                    diffs=[times_t[i]-times_t[i-1] for i in range(1,len(times_t))]
                    sec=int(sum(diffs)/len(diffs))
                elif len(times_t)==1:
                    sec=int(max(120, min(900, time.time()-times_t[-1])))
                else:
                    sec=180
            sec=max(120, min(int(sec), MAX_HORIZON_SECONDS))
            pred_ts = last_ts + sec
            combined = self.score_bandit(int(t), p)
            # adjust combined by pattern signals: if repeated strong then boost
            if repeated > 2:
                combined = min(1.0, combined + 0.05)
            in_cd = self.in_cooldown(t)
            thresh = CONF_THRESH.get(int(t), 0.55)
            recommended = (combined >= thresh) and (not in_cd)
            pending_id=None
            if recommended:
                ent = self.create_pending(t, sec, last_ts)
                pending_id = ent["id"]
            results.append({"target":int(t), "probability":round(float(p),3), "pred_seconds":int(sec), "pred_time_iso": datetime.fromtimestamp(pred_ts).strftime("%H:%M:%S"), "combined_score": round(float(combined),3), "recommended": recommended, "status": ("pending" if recommended else "not_recommended"), "pending_id": pending_id})
        recs=[r for r in results if r["recommended"]]
        if len(recs)==0:
            main = sorted(results, key=lambda x:(-x["combined_score"], x["pred_seconds"]))[0]; main["status"]="not_recommended"
        else:
            main = sorted(recs, key=lambda x:(-x["combined_score"], x["pred_seconds"]))[0]
        explain = compute_shap(feat) if HAS_SHAP else {}
        # compute suggested stake via Kelly using model probability and odds
        for r in results:
            p = r["probability"]
            b = r["target"] - 1.0
            r["kelly_frac"] = round(kelly_fraction(p, b), 4)
            r["suggested_stake_fraction"] = r["kelly_frac"] * 0.5  # conservative fraction of Kelly
        return {"all": results, "main": main, "explain": explain}
predictor = Predictor()

# ------------------ Background threads ----------------
def fetch_history_once():
    try:
        r = requests.get(URL_HISTORY, headers=HEADERS, timeout=8)
        r.raise_for_status()
        data = r.json()
        rounds = data.get("history", data.get("rounds", []))
        return rounds
    except Exception:
        return []

def poll_loop():
    while True:
        try:
            rounds = fetch_history_once()
            if rounds:
                buffer.extend(rounds)
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)

def train_loop():
    while True:
        time.sleep(TRAIN_INTERVAL)
        try:
            predictor.train_all()
            survival_models.train_from_buffer()
            ensemble.build_and_train()
        except Exception:
            traceback.print_exc()

poll_thread = threading.Thread(target=poll_loop, daemon=True)
train_thread = threading.Thread(target=train_loop, daemon=True)
monitor_thread = threading.Thread(target=predictor.monitor_loop, daemon=True)
poll_thread.start(); train_thread.start(); monitor_thread.start()

# ---------------- Visualization endpoints ----------------
def plot_history_png():
    df = buffer.dump_df()
    if df is None or len(df)==0:
        fig = plt.figure(figsize=(6,3)); plt.text(0.5,0.5,"No data",ha="center"); plt.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(df["ts"].apply(lambda t: datetime.fromtimestamp(t)), df["crash"], marker=".", linestyle="-", markersize=3)
        ax.set_ylabel("crash"); ax.set_xlabel("time"); fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------- Backtest endpoint (bankroll + performance metrics) ----------------
def backtest_strategy_fn(strategy_fn, df, initial_bank=1000):
    res = backtest_strategy(strategy_fn, df, initial_bank=initial_bank)
    return res

def backtest_strategy(strategy_fn, df, initial_bank=1000):
    bank = initial_bank
    trades=[]
    crashes = df["crash"].values
    n = len(crashes)
    for i in range(WINDOW_SIZE, n-1):
        state = {"index":i, "history": df.iloc[:i].copy()}
        target, stake = strategy_fn(state)
        if stake <= 0 or stake > bank:
            continue
        found=False
        for j in range(i, n):
            if crashes[j] >= target:
                payout = stake * (target - 1.0)
                bank += payout
                trades.append({"index":i,"target":target,"stake":stake,"result":"win","payout":payout})
                found=True; break
            if j==n-1 and not found:
                bank -= stake
                trades.append({"index":i,"target":target,"stake":stake,"result":"loss","payout": -stake})
        if bank <= 0:
            break
    wins = sum(1 for t in trades if t["result"]=="win")
    losses = sum(1 for t in trades if t["result"]=="loss")
    dd = max(0, initial_bank - min(initial_bank, min([initial_bank] + [initial_bank + sum([tr["payout"] for tr in trades[:i+1]]) for i in range(len(trades))])))
    return {"final_bank": bank, "pnl": bank-initial_bank, "wins": wins, "losses": losses, "trades": trades, "max_drawdown": dd}

# ---------------- Flask UI / API ----------------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prédicteur IA — Ultra Complet</title>
<style>
body{background:#071021;color:#fff;font-family:Inter,Arial;padding:12px}
.card{max-width:1000px;margin:auto;background:#091526;padding:18px;border-radius:12px}
.big{font-size:48px;color:#ffd54a}
.muted{color:#9fb0bd}
table{width:100%;border-collapse:collapse;margin-top:12px}
th,td{padding:8px;border-bottom:1px solid rgba(255,255,255,0.06);text-align:left}
button{background:#06b6d4;color:#042f3a;border:none;padding:8px 12px;border-radius:8px;cursor:pointer}
.bad{color:#ff6b6b}.good{color:#6bff9a}.pending{color:#ffd54a}
</style></head><body>
<div class="card">
<h2>🚀 Prédicteur IA — Ultra Complet</h2>
<p class="muted">Stats, patterns, ensemble, survival, calibration, drift, online learning, explainability, RL fallback, bankroll & plots.</p>
<button onclick="ask()">Demander prédiction</button>
<button onclick="bt()">Backtest simple</button>
<button onclick="dl()">Télécharger historique</button>
<button onclick="plot()">Voir graphique historique</button>
<div id="main" style="margin-top:12px"></div>
<div id="table"></div>
<h3>Prédictions récentes</h3><div id="recent"></div>
</div>
<script>
async function ask(){
  document.getElementById('main').innerText='Analyse en cours...';
  const r = await fetch('/predict');
  const j = await r.json();
  if(j.error){ document.getElementById('main').innerText = j.error; return; }
  const main=j.main;
  document.getElementById('main').innerHTML = '<div class="big">'+main.target+'x</div><div class="muted">Prévu: '+main.pred_time_iso+' (dans '+Math.round(main.pred_seconds/60)+' min) — score '+(main.combined_score*100).toFixed(1)+'%</div>';
  let html='<table><tr><th>Cible</th><th>Heure</th><th>Dans (min)</th><th>Proba</th><th>Score</th><th>Kelly</th><th>Status</th></tr>';
  for(const r of j.all){
    html += `<tr><td>${r.target}x</td><td>${r.pred_time_iso}</td><td>${(r.pred_seconds/60).toFixed(1)}</td><td>${(r.probability*100).toFixed(1)}%</td><td>${(r.combined_score*100).toFixed(1)}%</td><td>${(r.kelly_frac*100).toFixed(1)}%</td><td>${r.status}</td></tr>`;
  }
  html += '</table>';
  document.getElementById('table').innerHTML = html;
  const rr = await fetch('/recent_predictions'); const rj = await rr.json();
  let list = '<ul>';
  for(const p of rj.recent){ list += `<li>${p.requested_at_iso}: ${p.target}x — ${p.pred_time_iso} — ${p.status}</li>`; }
  list += '</ul>';
  document.getElementById('recent').innerHTML = list;
}
async function bt(){
  document.getElementById('main').innerText='Running backtest...';
  const r = await fetch('/backtest');
  const j = await r.json();
  if(j.error){ document.getElementById('main').innerText=j.error; return; }
  document.getElementById('main').innerText = 'Backtest: final bank='+j.final_bank+' pnl='+j.pnl+' wins='+j.wins+' losses='+j.losses;
}
async function dl(){ window.location.href='/download_history'; }
async function plot(){ window.open('/plot_history'); }
setInterval(async ()=>{ const rr=await fetch('/recent_predictions'); const rj=await rr.json(); let list='<ul>'; for(const p of rj.recent){ list += `<li>${p.requested_at_iso}: ${p.target}x — ${p.pred_time_iso} — ${p.status}</li>`; } list += '</ul>'; document.getElementById('recent').innerHTML = list; }, 8000);
</script></body></html>
"""

@app.route("/")
def home():
    return INDEX_HTML

@app.route("/predict", methods=["GET"])
def predict_route():
    res = predictor.predict()
    return jsonify(res)

@app.route("/recent_predictions", methods=["GET"])
def recent_route():
    out=[]
    with predictor.pending_lock:
        for p in list(predictor.recent)[:60]:
            pid = p.get("id")
            st = p.get("status","pending")
            if pid and pid in predictor.pending:
                st = predictor.pending[pid]["status"]
            out.append({"id": pid, "requested_at_iso": p["requested_at_iso"], "target": p["target"], "pred_time_iso": p["pred_time_iso"], "status": st})
    return jsonify({"recent": out})

@app.route("/backtest", methods=["GET"])
def backtest_route():
    df = buffer.dump_df()
    if df is None or len(df) < WINDOW_SIZE + 100:
        return jsonify({"error":"not enough history"})
    # simple test using evolutionary trainer too for strategy search
    res = backtest_strategy(lambda s: (2.0, 1.0), df, initial_bank=1000)
    evo = evo_trainer.train(df)
    return jsonify({"final_bank": res["final_bank"], "pnl": res["pnl"], "wins": res["wins"], "losses": res["losses"], "trades": len(res["trades"]), "evo_best": evo})

@app.route("/download_history", methods=["GET"])
def download_history():
    df = buffer.dump_df()
    if df is None or len(df)==0:
        return jsonify({"error":"no history"})
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    return send_file(io.BytesIO(csv_buf.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name="history.csv")

@app.route("/plot_history", methods=["GET"])
def plot_history():
    buf = plot_history_png()
    return Response(buf.getvalue(), mimetype="image/png")

@app.route("/metrics", methods=["GET"])
def metrics_route():
    crashes, timestamps = buffer.get_series()
    return jsonify({"buffer_len": len(crashes), "pending": len(predictor.pending), "cooldowns": predictor.cooldowns, "failure_counts": predictor.failure_counts, "drift_frozen": drift_detector.frozen})

@app.route("/report_outcome", methods=["POST"])
def report_outcome_route():
    j = request.json or {}
    t = float(j.get("target", 0))
    succ = bool(j.get("success", False))
    with predictor.pending_lock:
        for pid,e in predictor.pending.items():
            if e["target"]==t and e["status"]=="pending":
                e["status"]="success" if succ else "failure"
                predictor.pending[pid]=e
                return jsonify({"status":"ok","updated":pid})
    return jsonify({"status":"not_found"})

@app.route("/explain", methods=["GET"])
def explain():
    feat, last_ts = seq_features_for_last_window()
    return jsonify(compute_shap(feat) if HAS_SHAP else {"note":"shap not available"})

# ---------------- Start server ----------------
if __name__ == "__main__":
    print("Starting Ultra Bot. Optional libs:")
    print("TF", HAS_TF, "XGB", HAS_XGB, "LGB", HAS_LGB, "lifelines", HAS_LIFELINES, "river", HAS_RIVER, "shap", HAS_SHAP, "ruptures", HAS_RUPTURES, "mapie", HAS_MAPIE, "sb3", HAS_SB3)
    app.run(host="0.0.0.0", port=8080, debug=True)