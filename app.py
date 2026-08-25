import cv2
import threading
import os
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
BASE_RUNS_PATH = BASE_DIR / "runs" / "detect"
VIDEO_FOLDER_PATH = BASE_DIR / "testvideos"
DEFAULT_MODEL_PATH = BASE_DIR / "yolo11n.pt"
DATA_YAML_PATH = BASE_DIR / "data.yaml"

status = {"training": "Idle", "is_training": False}
stop_inference_event = threading.Event()

def get_yolo_model(path_or_name):
    from ultralytics import YOLO
    return YOLO(str(path_or_name))

def get_runs():
    """Scans the directory for all 'train' folders safely."""
    if not BASE_RUNS_PATH.exists():
        return []
    runs = [d.name for d in BASE_RUNS_PATH.iterdir() if d.is_dir()]
    return sorted(runs, reverse=True)

def get_videos():
    """Scans the testvideo folder for any playable video files."""
    if not VIDEO_FOLDER_PATH.exists():
        return []
    valid_exts = ('.mp4', '.avi', '.mkv', '.mov', '.webm')
    try:
        return [f.name for f in VIDEO_FOLDER_PATH.iterdir() if f.is_file() and f.name.lower().endswith(valid_exts)]
    except Exception:
        return []

@app.route('/')
def index():
    return render_template('index.html', runs=get_runs(), videos=get_videos())

@app.route('/get_runs_list')
def get_runs_list():
    return jsonify({"runs": get_runs()})

@app.route('/get_val_images')
def get_val_images():
    """Scans the selected run directory for all training metric charts and plots."""
    run_folder = request.args.get('run', '')
    if not run_folder:
        return jsonify([])
    
    target_dir = BASE_RUNS_PATH / run_folder
    if not target_dir.exists():
        return jsonify([])
        
    try:
        valid_extensions = ('.png', '.jpg', '.jpeg')
        files = [f.name for f in target_dir.iterdir() if f.is_file() and f.name.lower().endswith(valid_extensions)]
        return jsonify(sorted(files))
    except Exception:
        return jsonify([])

@app.route('/runs_file/<run_folder>/<filename>')
def runs_file(run_folder, filename):
    target_dir = BASE_RUNS_PATH / run_folder
    return send_from_directory(target_dir, filename)

@app.route('/video_feed')
def video_feed():
    # Force kill any remaining loop threads running in the background before spinning up a new stream
    stop_inference_event.set()
    stop_inference_event.clear()

    video_selection = request.args.get('path', '')
    run_folder = request.args.get('run', 'train')
    show_labels = request.args.get('labels') == 'true'
    box_width = int(request.args.get('width', 2))
    
    # Safely extract class checkbox filters from frontend query params
    show_class_0 = request.args.get('class0', 'true') == 'true'
    show_class_1 = request.args.get('class1', 'true') == 'true'
    
    # Map active filters to integer indexes for YOLO
    active_classes = []
    if show_class_0: active_classes.append(0)
    if show_class_1: active_classes.append(1)

    model_path = BASE_RUNS_PATH / run_folder / "weights" / "best.pt"
    
    try:
        model = get_yolo_model(model_path)
    except Exception:
        model = get_yolo_model(DEFAULT_MODEL_PATH)

    if video_selection.isdigit():
        source = int(video_selection)
    else:
        source = VIDEO_FOLDER_PATH / video_selection

    def generate():
        if isinstance(source, Path) and not source.exists():
            return
            
        cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        while cap.isOpened():
            if stop_inference_event.is_set():
                break
                
            success, frame = cap.read()
            if not success: 
                break
            
            # If both checkboxes are unchecked, we tell YOLO to look for class index -1 (which catches nothing)
            # This cleanly hides all boxes without throwing an exception or crashing the stream thread
            filter_classes = active_classes if len(active_classes) > 0 else [-1]
            
            try:
                results = model(frame, conf=0.4, classes=filter_classes)
                annotated_frame = results[0].plot(
                    labels=show_labels, 
                    line_width=box_width
                )
            except Exception:
                # Fallback if specific weight index format differs
                results = model(frame, conf=0.4)
                annotated_frame = results[0].plot(labels=show_labels, line_width=box_width)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    stop_inference_event.set()
    return jsonify({"status": "Stream stopped"})

@app.route('/start-train', methods=['POST'])
def start_train():
    if status["is_training"]:
        return jsonify({"message": "Training already in progress!"}), 400
    
    data = request.json
    def run_training():
        status["is_training"] = True
        status["training"] = "In Progress..."
        try:
            model = get_yolo_model(DEFAULT_MODEL_PATH)
            model.train(
                data=str(DATA_YAML_PATH),
                epochs=int(data['epochs']), 
                device=data['device'],
                batch=int(data['batch']),
                imgsz=int(data['imgsz']),
                optimizer=data['optimizer'],
                lr0=float(data['lr0']),
                workers=0,
            )
            status["training"] = "Completed"
        except Exception as e:
            status["training"] = f"Error: {str(e)}"
        finally:
            status["is_training"] = False

    threading.Thread(target=run_training).start()
    return jsonify({"message": "Training started."})

@app.route('/status')
def get_status():
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
