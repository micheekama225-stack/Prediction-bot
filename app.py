from flask import Flask, jsonify
import requests

app = Flask(__name__)

@app.route("/history")
def history():
    url = "https://crash-gateway-grm-cr.100hp.app/history"
    headers = {
        "accept": "application/json, text/plain, */*",
        "origin": "https://1play.gamedev-tech.cc",
        "referer": "https://1play.gamedev-tech.cc/",
        "customer-id": "077dee8d-c923-4c02-9bee-757573662e69",
        "session-id": "1f56b906-1fe1-4cc8-add1-34222b858a7e",
        "user-agent": "Mozilla/5.0"
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()  # lève une erreur si status != 200
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
