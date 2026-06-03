from flask import Flask, render_template, request
from summarizer import generate_summary

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    summary = ""
    accuracy = 0
    input_text = ""

    if request.method == "POST":

        input_text = request.form.get("text", "").strip()

        if len(input_text.split()) < 10:

            summary = "Please enter more text."

        else:

            summary, accuracy = generate_summary(
                input_text
            )

    return render_template(
        "index.html",
        summary=summary,
        accuracy=accuracy,
        input_text=input_text
    )


if __name__ == "__main__":
    app.run(debug=True)