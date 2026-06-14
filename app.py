from flask import Flask,render_template,Response
import cv2
app=Flask(__name__)
from detect_face_mask import edit_frames


def generate_frames():
    camera =cv2.VideoCapture(0)
    while True:   
        success,frame1=camera.read()
        if not success:
            break
        else:
            frame1 = edit_frames(frame1)
            ret,buffer=cv2.imencode('.jpg',frame1)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)