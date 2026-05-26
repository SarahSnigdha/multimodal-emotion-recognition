# ============================================================
# MAIN.PY
# ============================================================

import os

from flask import (
    Flask,
    render_template,
    request
)

from models.speech_pipeline.speech_test import (
    speech_predict
)

from models.text_pipeline.text_test import (
    text_predict
)

from models.fusion_pipeline.fusion_test import (
    fusion_predict
)

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER

# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    speech_result = None

    text_result = None

    fusion_result = None

    error = None

    if request.method == "POST":

        text_input = (
            request.form.get(
                "text_input"
            )
        )

        audio_file = (
            request.files.get(
                "audio_file"
            )
        )

        audio_path = None

        # ----------------------------------------------------
        # SAVE AUDIO
        # ----------------------------------------------------

        if (
            audio_file
            and audio_file.filename != ""
        ):

            audio_path = os.path.join(
                app.config[
                    "UPLOAD_FOLDER"
                ],
                audio_file.filename
            )

            audio_file.save(
                audio_path
            )

        # ----------------------------------------------------
        # NO INPUT
        # ----------------------------------------------------

        if (
            not audio_path
            and not text_input
        ):

            error = (
                "Please upload audio, "
                "enter text, or both."
            )

            return render_template(
                "index.html",
                error=error
            )

        # ----------------------------------------------------
        # AUDIO ONLY
        # ----------------------------------------------------

        if (
            audio_path
            and not text_input
        ):

            speech_result = (
                speech_predict(
                    audio_path
                )
            )

        # ----------------------------------------------------
        # TEXT ONLY
        # ----------------------------------------------------

        elif (
            text_input
            and not audio_path
        ):

            text_result = (
                text_predict(
                    text_input
                )
            )

        # ----------------------------------------------------
        # AUDIO + TEXT
        # ----------------------------------------------------

        else:

            speech_result = (
                speech_predict(
                    audio_path
                )
            )

            text_result = (
                text_predict(
                    text_input
                )
            )

            fusion_result = (
                fusion_predict(
                    audio_path=audio_path,
                    text=text_input
                )
            )

    return render_template(
        "index.html",
        speech_result=speech_result,
        text_result=text_result,
        fusion_result=fusion_result,
        error=error
    )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )