from pathlib import Path

from flask import Flask, Response, request, send_file
from ollamafreeapi import OllamaFreeAPI

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "llama3.2:latest"

app = Flask(__name__)
client = OllamaFreeAPI()


@app.after_request
def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index() -> Response:
    return send_file(BASE_DIR / "index.html")


@app.route("/chat", methods=["GET", "POST", "OPTIONS"])
def chat() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        model = (payload.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    else:
        prompt = (request.args.get("prompt") or "").strip()
        model = (request.args.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if not prompt:
        return Response("Bitte einen Prompt eingeben.", status=400, mimetype="text/plain")

    def generate():
        try:
            for chunk in client.stream_chat(prompt, model):
                if chunk:
                    yield chunk
        except Exception as exc:
            yield f"\n[Fehler] {exc}\n"

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
