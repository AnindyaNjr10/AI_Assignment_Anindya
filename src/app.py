"""Task 6 Option B - small Flask web interface.   Run: python src/app.py  ->  http://127.0.0.1:5000"""
from flask import Flask, render_template_string, request

from predict import load_model, predict_ticket
from responses import suggested_response

app = Flask(__name__)
model = load_model()

PAGE = """
<!doctype html><title>Customer Support Classifier</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:640px;margin:48px auto;padding:0 16px;color:#222}
 h1{text-align:center} textarea{width:100%;height:110px;padding:10px;font-size:16px;box-sizing:border-box}
 button{display:block;margin:14px auto;padding:10px 28px;font-size:16px;cursor:pointer}
 .result{background:#f3f6fb;border-left:5px solid #4C72B0;padding:14px 18px;margin-top:20px}
 .bar{background:#dde3ee;border-radius:6px;height:12px;margin-top:6px}
 .bar div{background:#4C72B0;height:12px;border-radius:6px}
</style>
<h1>Customer Support Classifier</h1>
<form method="post">
 <label for="t"><b>Ticket Description:</b></label>
 <textarea id="t" name="ticket" placeholder="e.g. The application is very slow today">{{ text }}</textarea>
 <button type="submit">Predict</button>
</form>
{% if category %}
<div class="result">
 <p><b>Predicted Category:</b> {{ category }}</p>
 <p><b>Confidence:</b> {{ '%.1f'|format(conf) }}%</p>
 <div class="bar"><div style="width:{{ conf }}%"></div></div>
 <p><b>Suggested Response:</b><br>{{ reply }}</p>
</div>
{% elif error %}<p style="color:#b00">{{ error }}</p>{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"text": "", "category": None, "conf": 0, "reply": "", "error": None}
    if request.method == "POST":
        text = request.form.get("ticket", "").strip()
        ctx["text"] = text
        if not text:
            ctx["error"] = "Please enter a ticket description."
        else:
            ctx["category"], ctx["conf"] = predict_ticket(model, text)
            ctx["reply"] = suggested_response(ctx["category"])
    return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    app.run(debug=False)
