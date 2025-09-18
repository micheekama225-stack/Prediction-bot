import requests
from collections import deque
from datetime import datetime
import time
import numpy as np
import pandas as pd

import requests
from collections import deque
from datetime import datetime
import time
import numpy as np
import pandas as pd

# IA et ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Analyse
from lifelines import KaplanMeierFitter
import shap

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Flask API
from flask import Flask, jsonify, render_template_string

# Logs
from loguru import logger

# --------------------
# CONFIG
# --------------------
URL_HISTORY = "https://crash-gateway-grm-cr.100hp.app/history"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://1play.gamedev-tech.cc",
    "referer": "https://1play.gamedev-tech.cc/",
    "customer-id": "077dee8d-c923-4c02-9bee-757573662e69",
    "session-id": "1f56b906-1fe1-4cc8-add1-34222b858a7e",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# --------------------
# BOT IA
# --------------------
class CrashBot:
    def __init__(self, window_size=20):
        self.history = deque(maxlen=10000)     # crash points
        self.timestamps = deque(maxlen=10000)  # heures des rounds
        self.window_size = window_size

        # Plusieurs modèles
        self.models = {
            "rf": RandomForestClassifier(),
            "gb": GradientBoostingClassifier(),
            "logreg": LogisticRegression(max_iter=500),
            "svm": SVC(probability=True),
            "mlp": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500)
        }
        self.trained = False

    def fetch_history(self):
        """Récupère l'historique depuis l'API"""
        try:
            r = requests.get(URL_HISTORY, headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()
            rounds = data.get("history", data.get("rounds", []))
            return rounds
        except Exception as e:
            logger.error(f"Erreur API: {e}")
            return []

    def update_history(self):
        """Met à jour l'historique avec les derniers tours"""
        rounds = self.fetch_history()
        for rd in rounds:
            cp = rd.get("crash_point")
            ts = rd.get("timestamp")
            if cp is not None:
                self.history.append(cp)
                if ts is None:
                    ts = int(time.time())
                self.timestamps.append(int(ts))

    def prepare_dataset(self):
        """Crée le dataset pour l'entraînement"""
        X, y = [], []
        hist = list(self.history)
        for i in range(len(hist) - self.window_size - 1):
            X.append(hist[i:i+self.window_size])
            y.append(1 if hist[i+self.window_size] >= 3 else 0)
        return np.array(X), np.array(y)

    def train(self):
        """Entraîne tous les modèles IA"""
        if len(self.history) < self.window_size + 50:
            return
        X, y = self.prepare_dataset()
        if len(set(y)) > 1:
            for name, model in self.models.items():
                try:
                    model.fit(X, y)
                    logger.info(f"Modèle {name} entraîné")
                except Exception as e:
                    logger.error(f"Erreur entraînement {name}: {e}")
            self.trained = True

    def round_to_targets(self, value, targets=[2, 3, 5, 10]):
        """Arrondit la cote au plus proche de 2, 3, 5 ou 10"""
        return min(targets, key=lambda t: abs(t - value))

    def predict_next(self):
        """Prédit le prochain événement"""
        if len(self.history) < self.window_size:
            return {"prediction": "Pas assez de données", "heure": datetime.now().strftime("%H:%M")}

        last_seq = list(self.history)[-self.window_size:]
        results = {}

        if self.trained:
            for name, model in self.models.items():
                try:
                    proba = model.predict_proba([last_seq])[0][1]
                    results[name] = proba
                except Exception:
                    results[name] = 0.5
        else:
            results = {name: 0.5 for name in self.models}

        avg_conf = np.mean(list(results.values()))

        # Détermination de la cote arrondie
        last_val = self.history[-1]
        predicted_coef = self.round_to_targets(last_val * (1 + avg_conf))

        # Estimation du temps futur
        three_x_times = [t for (c, t) in zip(self.history, self.timestamps) if c >= 3]
        if len(three_x_times) >= 2:
            diffs = [three_x_times[i] - three_x_times[i-1] for i in range(1, len(three_x_times))]
            avg_interval = sum(diffs) / len(diffs)
            predicted_ts = three_x_times[-1] + avg_interval
        else:
            predicted_ts = int(time.time()) + 180

        dt = datetime.fromtimestamp(predicted_ts)
        return {
            "prediction": f"≥ {predicted_coef}x",
            "heure": dt.strftime("%H:%M"),
            "confiance": round(avg_conf, 2),
            "details": results
        }

# --------------------
# FLASK
# --------------------
bot = CrashBot(window_size=20)
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Prédicteur CrashBot</title>
  <style>
    body { text-align:center; background:#111; color:#fff; font-family:Arial; margin-top:80px; }
    button { padding:15px 30px; font-size:18px; border:none; border-radius:10px; background:#28a745; color:#fff; cursor:pointer; }
    button:hover { background:#218838; }
    #result { margin-top:50px; font-size:64px; font-weight:bold; color:#ffcc00; }
    #time { margin-top:10px; font-size:28px; color:#bbb; }
    #conf { margin-top:10px; font-size:20px; color:#88f; }
  </style>
</head>
<body>
  <h1>🚀 CrashBot – Prédicteur de cotes</h1>
  <button onclick="getPrediction()">Demander prédiction</button>
  <div id="result"></div>
  <div id="time"></div>
  <div id="conf"></div>

  <script>
    async function getPrediction() {
      document.getElementById("result").innerText = "Analyse en cours...";
      document.getElementById("time").innerText = "";
      document.getElementById("conf").innerText = "";
      const resp = await fetch("/predict");
      const d = await resp.json();
      document.getElementById("result").innerText = d.prediction;
      document.getElementById("time").innerText = "Prévu à : " + d.heure;
      document.getElementById("conf").innerText = "Confiance : " + d.confiance;
    }
  </script>
</body>
</html>
"""

@app.route("/")
def home():
    return HTML_PAGE

@app.route("/predict", methods=["GET"])
def predict():
    bot.update_history()
    bot.train()
    return jsonify(bot.predict_next())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)￼Enter# IA et ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Analyse
from lifelines import KaplanMeierFitter
import shap

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Flask API
from flask import Flask, jsonify, render_template_string

# Logs
from loguru import logger

# --------------------
# CONFIG
# --------------------
URL_HISTORY = "https://crash-gateway-grm-cr.100hp.app/history"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://1play.gamedev-tech.cc",
    "referer": "https://1play.gamedev-tech.cc/",
    "customer-id": "077dee8d-c923-4c02-9bee-757573662e69",
    "session-id": "1f56b906-1fe1-4cc8-add1-34222b858a7e",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# --------------------
# BOT IA
# --------------------
class CrashBot:
    def __init__(self, window_size=20):
        self.history = deque(maxlen=10000)     # crash points
        self.timestamps = deque(maxlen=10000)  # heures des rounds
        self.window_size = window_size

        # Plusieurs modèles
        self.models = {
            "rf": RandomForestClassifier(),
            "gb": GradientBoostingClassifier(),
            "logreg": LogisticRegression(max_iter=500),
            "svm": SVC(probability=True),
            "mlp": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500)
        }
        self.trained = False

    def fetch_history(self):
        """Récupère l'historique depuis l'API"""
        try:
            r = requests.get(URL_HISTORY, headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()
            rounds = data.get("history", data.get("rounds", []))
            return rounds
        except Exception as e:
            logger.error(f"Erreur API: {e}")
            return []

    def update_history(self):
        """Met à jour l'historique avec les derniers tours"""
  rounds = self.fetch_history()
        for rd in rounds:
            cp = rd.get("crash_point")
            ts = rd.get("timestamp")
            if cp is not None:
                self.history.append(cp)
                if ts is None:
                    ts = int(time.time())
                self.timestamps.append(int(ts))

    def prepare_dataset(self):
        """Crée le dataset pour l'entraînement"""
        X, y = [], []
        hist = list(self.history)
        for i in range(len(hist) - self.window_size - 1):
            X.append(hist[i:i+self.window_size])
            y.append(1 if hist[i+self.window_size] >= 3 else 0)
        return np.array(X), np.array(y)

    def train(self):
        """Entraîne tous les modèles IA"""
        if len(self.history) < self.window_size + 50:
            return
        X, y = self.prepare_dataset()
        if len(set(y)) > 1:
            for name, model in self.models.items():
                try:
                    model.fit(X, y)
                    logger.info(f"Modèle {name} entraîné")
                except Exception as e:
                    logger.error(f"Erreur entraînement {name}: {e}")
            self.trained = True

    def round_to_targets(self, value, targets=[2, 3, 5, 10]):
        """Arrondit la cote au plus proche de 2, 3, 5 ou 10"""
        return min(targets, key=lambda t: abs(t - value))

    def predict_next(self):
        """Prédit le prochain événement"""
        if len(self.history) < self.window_size:
            return {"prediction": "Pas assez de données", "heure": datetime.now().strftime("%H:%M")}

        last_seq = list(self.history)[-self.window_size:]
        results = {}

        if self.trained:
            for name, model in self.models.items():
                try:
                    proba = model.predict_proba([last_seq])[0][1]
                    results[name] = proba
                except Exception:
                    results[name] = 0.5
        else:
            results = {name: 0.5 for name in self.models}

        avg_conf = np.mean(list(results.values()))

        # Détermination de la cote arrondie
        last_val = self.history[-1]
        predicted_coef = self.round_to_targets(last_val * (1 + avg_conf))

        # Estimation du temps futur
        three_x_times = [t for (c, t) in zip(self.history, self.timestamps) if c >= 3]
        if len(three_x_times) >= 2:
            diffs = [three_x_times[i] - three_x_times[i-1] for i in range(1, len(three_x_times))]
            avg_interval = sum(diffs) / len(diffs)
