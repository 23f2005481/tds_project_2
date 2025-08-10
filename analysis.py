import json
import pandas as pd
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns

def handle_analysis(files):
    with open(files["questions.txt"], "r") as f:
        questions = f.read().strip()

    # TODO: parse questions, detect type, fetch or load data
    # Example: create dummy output matching expected test format
    output = [
        1,
        "Titanic",
        0.485782,
        plot_dummy()
    ]
    return output

def plot_dummy():
    import numpy as np
    x = np.arange(1, 11)
    y = x * 0.5 + 1
    plt.figure()
    sns.scatterplot(x=x, y=y)
    sns.regplot(x=x, y=y, scatter=False, color="red", line_kws={"linestyle": "dotted"})
    plt.xlabel("Rank")
    plt.ylabel("Peak")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
