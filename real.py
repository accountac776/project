import threading
from main import YOLOv8Live
from updatedonMyRightLeft import VoiceRecorder

def run_camera_module(yolo_live):
    yolo_live.run()

def run_audio_module(yolo_live):
    recorder = VoiceRecorder(yolo_live)
    recorder.run()

if __name__ == "__main__":
    print("Parsing arguments...")
    args = YOLOv8Live.parse_arguments()
    yolo_live = YOLOv8Live(front_resolution=args.front_resolution, back_resolution=args.back_resolution)

    print("Starting camera and audio threads...")
    camera_thread = threading.Thread(target=run_camera_module, args=(yolo_live,))
    audio_thread = threading.Thread(target=run_audio_module, args=(yolo_live,))

    camera_thread.start()
    audio_thread.start()

    camera_thread.join()
    audio_thread.join()