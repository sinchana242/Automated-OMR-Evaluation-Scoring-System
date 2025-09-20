import os
from flask import Flask, request, render_template
from omr_grader import grade_omr

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Answer key: question_index -> correct choice index (0‑based)
ANSWER_KEY = {
    0: 1,
    1: 3,
    2: 0,
    3: 2,
    4: 1,
    5: 0,
    6: 3,
    7: 2,
    8: 1,
    9: 0
}

NUM_QUESTIONS = len(ANSWER_KEY)
NUM_CHOICES = 4

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    result_img_filename = None
    if request.method == 'POST':
        if 'omr_image' not in request.files:
            return "No file key omr_image", 400
        file = request.files['omr_image']
        if file.filename == '':
            return "No selected file", 400
        # save uploaded
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        if not os.path.exists(RESULT_FOLDER):
            os.makedirs(RESULT_FOLDER)
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            detected, score, vis_image = grade_omr(filepath, ANSWER_KEY, NUM_QUESTIONS, NUM_CHOICES)
        except Exception as e:
            return f"Error: {e}", 500

        # save visualization
        result_filename = 'result_' + file.filename
        result_filepath = os.path.join(RESULT_FOLDER, result_filename)
        # vis_image is BGR (OpenCV), convert to RGB for PIL or cv2.imwrite accepts BGR but okay
        from cv2 import imwrite
        imwrite(result_filepath, vis_image)

        result_img_filename = result_filename

    return render_template('index.html', score=score, result_image=result_img_filename)

if __name__ == '__main__':
    app.run(debug=True)
