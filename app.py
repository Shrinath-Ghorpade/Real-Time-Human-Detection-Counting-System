from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import imutils
import os
from werkzeug.utils import secure_filename
from detect import detect

# Flask setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"

# Ensure folders exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


# ------------------- IMAGE UPLOAD -------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    image = imutils.resize(image, width=min(800, image.shape[1]))
    result = detect(image)

    result_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    cv2.imwrite(result_path, result)

    return render_template("result.html",
                           uploaded_image=filepath,
                           result_image=result_path)


# ------------------- VIDEO UPLOAD -------------------
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return redirect(url_for("index"))

    file = request.files["video"]
    if file.filename == "":
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Process video
    cap = cv2.VideoCapture(filepath)
    result_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=800)
        frame = detect(frame)

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(result_path,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  10, (width, height))

        out.write(frame)

    cap.release()
    if out:
        out.release()

    return render_template("result.html",
                           uploaded_image=filepath,
                           result_image=result_path)


# ------------------- CAMERA STREAM -------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=800)
            frame = detect(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/camera')
def camera():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__": 
    app.run(debug=True)