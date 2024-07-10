import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv
import speech_recognition as sr
import os
from gtts import gTTS

class YOLOv8Live:
    def __init__(self, front_resolution, back_resolution):
        print(f"Initializing YOLOv8Live with front_resolution={front_resolution} and back_resolution={back_resolution}")
        self.front_frame_width, self.front_frame_height = front_resolution
        self.back_frame_width, self.back_frame_height = back_resolution
        self.cap_front = cv2.VideoCapture(0)  # Front camera
        self.cap_back = cv2.VideoCapture(1)   # Back camera
        self.cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, self.front_frame_width)
        self.cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, self.front_frame_height)
        self.cap_back.set(cv2.CAP_PROP_FRAME_WIDTH, self.back_frame_width)
        self.cap_back.set(cv2.CAP_PROP_FRAME_HEIGHT, self.back_frame_height)
        
        self.model = YOLO("yolov8n.pt")
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
        
        self.zone_polygon_front_left = np.array([
            [0, 0],
            [0.50 * self.front_frame_width, 0],
            [0.50 * self.front_frame_width, self.front_frame_height],
            [0, self.front_frame_height]
        ]).astype(int)
        self.zone_polygon_front_right = np.array([
            [0.50 * self.front_frame_width, 0],
            [self.front_frame_width, 0],
            [self.front_frame_width, self.front_frame_height],
            [0.50 * self.front_frame_width, self.front_frame_height]
        ]).astype(int)
        self.zone_polygon_back_left = np.array([
            [0, 0],
            [0.50 * self.back_frame_width, 0],
            [0.50 * self.back_frame_width, self.back_frame_height],
            [0, self.back_frame_height]
        ]).astype(int)
        self.zone_polygon_back_right = np.array([
            [0.50 * self.back_frame_width, 0],
            [self.back_frame_width, 0],
            [self.back_frame_width, self.back_frame_height],
            [0.50 * self.back_frame_width, self.back_frame_height]
        ]).astype(int)

        self.zone_front_left = sv.PolygonZone(polygon=self.zone_polygon_front_left)
        self.zone_annotator_front_left = sv.PolygonZoneAnnotator(zone=self.zone_front_left, color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2)
        self.zone_front_right = sv.PolygonZone(polygon=self.zone_polygon_front_right)
        self.zone_annotator_front_right = sv.PolygonZoneAnnotator(zone=self.zone_front_right, color=sv.Color.BLUE, thickness=2, text_thickness=4, text_scale=2)
        self.zone_back_left = sv.PolygonZone(polygon=self.zone_polygon_back_left)
        self.zone_annotator_back_left = sv.PolygonZoneAnnotator(zone=self.zone_back_left, color=sv.Color.GREEN, thickness=2, text_thickness=4, text_scale=2)
        self.zone_back_right = sv.PolygonZone(polygon=self.zone_polygon_back_right)
        self.zone_annotator_back_right = sv.PolygonZoneAnnotator(zone=self.zone_back_right, color=sv.Color.YELLOW, thickness=2, text_thickness=4, text_scale=2)
        
        self.detections = None

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="YOLOv8 live")
        parser.add_argument("--front-resolution", default=[720, 400], nargs=2, type=int)
        parser.add_argument("--back-resolution", default=[1300, 720], nargs=2, type=int)
        args = parser.parse_args()
        return args

    def run(self):
        print("Starting YOLOv8Live run method...")
        while True:
            ret_front, frame_front = self.cap_front.read()
            ret_back, frame_back = self.cap_back.read()
            if not ret_front or not ret_back:
                print("Failed to read from cameras")
                break

            frame_front = self.process_frame(frame_front, 'front')
            frame_back = self.process_frame(frame_back, 'back')

            cv2.imshow("Front Camera", frame_front)
            cv2.imshow("Back Camera", frame_back)
            
            if cv2.waitKey(30) == 27:
                print("Exiting...")
                break

        self.cap_front.release()
        self.cap_back.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, camera_position):
        result = self.model(frame)[0]
        self.detections = sv.Detections.from_ultralytics(result)
        self.detections = self.detections[self.detections.class_id != 0]

        labels = [
            f"{self.model.names[class_id]} :: {conf:0.2f}"
            for xyxy, mask, conf, class_id, tracker_id, data in self.detections
        ]

        frame = self.bounding_box_annotator.annotate(scene=frame, detections=self.detections)
        frame = self.label_annotator.annotate(scene=frame, detections=self.detections, labels=labels)

        if camera_position == 'front':
            self.zone_front_left.trigger(detections=self.detections)
            frame = self.zone_annotator_front_left.annotate(scene=frame)
            self.zone_front_right.trigger(detections=self.detections)
            frame = self.zone_annotator_front_right.annotate(scene=frame)
        elif camera_position == 'back':
            self.zone_back_left.trigger(detections=self.detections)
            frame = self.zone_annotator_back_left.annotate(scene=frame)
            self.zone_back_right.trigger(detections=self.detections)
            frame = self.zone_annotator_back_right.annotate(scene=frame)

        return frame

    def get_zone_detections(self, zone_polygon, detections):
        zone_detections = []
        result_list = []
        
        for xyxy, mask, conf, class_id, tracker_id, data in detections:
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            tlx, tly = zone_polygon[0]
            brx, bry = zone_polygon[2]
            
            if tlx <= cx <= brx and tly <= cy <= bry:
                zone_detections.append(self.model.names[class_id])
        
        unique_classes = list(set(zone_detections))
        for cls in unique_classes:
            count = zone_detections.count(cls)
            result_list.append(f"{count} {cls}")
        
        return result_list

    def front_left_detections(self):
        return self.get_zone_detections(self.zone_polygon_front_left, self.detections)

    def front_right_detections(self):
        return self.get_zone_detections(self.zone_polygon_front_right, self.detections)

    def back_left_detections(self):
        return self.get_zone_detections(self.zone_polygon_back_left, self.detections)

    def back_right_detections(self):
        return self.get_zone_detections(self.zone_polygon_back_right, self.detections)

    def detection(self):
        print("Starting detection method...")
        # Extract detected objects in each zone
        front_left_result = [item.split(' ', 1)[1] for item in self.front_left_detections()]
        print(f'FRONT LEFT = {front_left_result}')
        back_left_result = [item.split(' ', 1)[1] for item in self.back_left_detections()]
        print(f'BACK LEFT = {back_left_result}')
        front_right_result = [item.split(' ', 1)[1] for item in self.front_right_detections()]
        print(f'FRONT RIGHT = {front_right_result}')
        back_right_result = [item.split(' ', 1)[1] for item in self.back_right_detections()]
        print(f'BACK RIGHT = {back_right_result}')
        names = list(self.model.names.values())
        print(f'Object names: {names}')

        # Recognize speech from audio file
        r = sr.Recognizer()
        with sr.AudioFile("messageOfUser.wav") as source:
            audio = r.record(source)

        try:
            recognized_text = r.recognize_google(audio).lower()
            print(f"Recognized Text: {recognized_text}")
        except sr.UnknownValueError:
            print("Could not understand audio")
            self._speak("Sorry, I did not understand that.")
            return
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            self._speak("Sorry, there was an error with the speech recognition service.")
            return

        # Check for zone-specific commands
        if "front left" in recognized_text:
            self._speak_zone_detections(self.front_left_detections(), "front left")
        elif "front right" in recognized_text:
            self._speak_zone_detections(self.front_right_detections(), "front right")
        elif "back left" in recognized_text:
            self._speak_zone_detections(self.back_left_detections(), "back left")
        elif "back right" in recognized_text:
            self._speak_zone_detections(self.back_right_detections(), "back right")
        else:
            # Check if specific objects are mentioned
            found = False
            for x in names:
                if x in recognized_text:
                    found = True
                    location = self._find_object_location(x, front_left_result, front_right_result, back_left_result, back_right_result)
                    if location:
                        self._speak(f'The {x} is in the {location}')
                    else:
                        self._speak(f'No {x} is detected')
            if not found:
                self._speak("I did not recognize the object or location.")

    def _speak_zone_detections(self, detections, zone_name):
        if detections:
            self._speak(f'There is a: {", ".join(detections)} in the {zone_name}')
        else:
            self._speak(f"No objects detected in the {zone_name}.")

    def _find_object_location(self, object_name, front_left_result, front_right_result, back_left_result, back_right_result):
        if object_name in front_left_result:
            return "front left"
        elif object_name in front_right_result:
            return "front right"
        elif object_name in back_left_result:
            return "back left"
        elif object_name in back_right_result:
            return "back right"
        return None



    def _speak(self, text):
        language = 'en'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("messageOfComputer.mp3")
        os.system("start messageOfComputer.mp3")

if __name__ == "__main__":
    args = YOLOv8Live.parse_arguments()
    yolo_live = YOLOv8Live(front_resolution=args.front_resolution, back_resolution=args.back_resolution)
    yolo_live.run()