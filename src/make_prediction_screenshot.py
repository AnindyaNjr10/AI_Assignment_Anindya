"""Renders the test-case results as screenshots/prediction.png"""
import io, contextlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict import load_model, run_tests
from preprocessing import ROOT
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    run_tests(load_model())
text = buf.getvalue()
fig, ax = plt.subplots(figsize=(13, 4.6)); ax.axis("off")
ax.set_title("New-ticket predictions (python src/predict.py --test)", loc="left", fontweight="bold")
ax.text(0, 1, text, family="monospace", fontsize=8.3, va="top")
fig.savefig(ROOT / "screenshots" / "prediction.png", dpi=130, bbox_inches="tight")
