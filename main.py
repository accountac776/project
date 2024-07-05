import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv
import speech_recognition as sr
import os
from gtts import gTTS

class YOLOv8Live:
    ZONE_POLYGON_LEFT = np.array([
        [0, 0],
        [0.50, 0],
        [0.50, 1],
        [0, 1]
    ])
    
    ZONE_POLYGON_RIGHT = np.array([
        [0.50, 0],
        [1, 0],
        [1, 1],
        [0.50, 1]
    ])
    
    def __init__(self, webcam_resolution):
        self.frame_width, self.frame_height = webcam_resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        self.model = YOLO("yolov8n.pt")
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
        
        self.zone_polygon_left = (self.ZONE_POLYGON_LEFT * np.array(webcam_resolution)).astype(int)
        self.zone_left = sv.PolygonZone(polygon=self.zone_polygon_left)
        self.zone_annotator_left = sv.PolygonZoneAnnotator(zone=self.zone_left, color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2)
        
        self.zone_polygon_right = (self.ZONE_POLYGON_RIGHT * np.array(webcam_resolution)).astype(int)
        self.zone_right = sv.PolygonZone(polygon=self.zone_polygon_right)
        self.zone_annotator_right = sv.PolygonZoneAnnotator(zone=self.zone_right, color=sv.Color.BLUE, thickness=2, text_thickness=4, text_scale=2)
        
        self.detections = None

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="YOLOv8 live")
        parser.add_argument("--webcam-resolution", default=[1200, 720], nargs=2, type=int)
        args = parser.parse_args()
        return args

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            result = self.model(frame)[0]
            self.detections = sv.Detections.from_ultralytics(result)
            self.detections = self.detections[self.detections.class_id != 0]

            labels = [
                f"{self.model.names[class_id]} :: {conf:0.2f}"
                for xyxy, mask, conf, class_id, tracker_id, data in self.detections
            ]

            frame = self.bounding_box_annotator.annotate(scene=frame, detections=self.detections)
            frame = self.label_annotator.annotate(scene=frame, detections=self.detections, labels=labels)

            self.zone_left.trigger(detections=self.detections)
            frame = self.zone_annotator_left.annotate(scene=frame)

            self.zone_right.trigger(detections=self.detections)
            frame = self.zone_annotator_right.annotate(scene=frame)

            cv2.imshow("yolov8", frame)
            
            if cv2.waitKey(30) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def left_detections(self):
        left_detections = []
        result_list = []
        
        if self.detections is not None:
            for xyxy, mask, conf, class_id, tracker_id, data in self.detections:
                x1, y1, x2, y2 = xyxy
                tlx, tly = self.zone_polygon_left[0]
                brx, bry = self.zone_polygon_left[2]
                
                if x1 >= tlx and y1 >= tly and x2 <= brx and y2 <= bry:
                    left_detections.append(self.model.names[class_id])
        
        unique_classes = list(set(left_detections))
        for cls in unique_classes:
            count = left_detections.count(cls)
            result_list.append(f"{count} {cls}")
        
        return result_list

    def right_detections(self):
        right_detections = []
        result_list = []
        
        if self.detections is not None:
            for xyxy, mask, conf, class_id, tracker_id, data in self.detections:
                x1, y1, x2, y2 = xyxy
                tlx, tly = self.zone_polygon_left[0]
                brx, bry = self.zone_polygon_left[2]
                
                if x1 >= tlx and y1 >= tly and x2 <= brx and y2 <= bry:
                    right_detections.append(self.model.names[class_id])
        
        unique_classes = list(set(right_detections))
        for cls in unique_classes:
            count = right_detections.count(cls)
            result_list.append(f"{count} {cls}")
        
        return result_list
    
    def detection(self):
        r = sr.Recognizer()
        with sr.AudioFile("messageOfUser.wav") as source:
            audio = r.record(source)

        recognized_text = r.recognize_google(audio) if audio else ""

        if "left" in recognized_text and "right" not in recognized_text:
            left_detections = self.left_detections()
            if left_detections:
                self._speak(f'There is a: {", ".join(left_detections)}')
            else:
                self._speak("No objects detected on the left.")
        elif "right" in recognized_text and "left" not in recognized_text:
            right_detections = self.right_detections()
            if right_detections:
                self._speak(f'There is a: {", ".join(right_detections)}')
            else:
                self._speak("No objects detected on the right.")
        else:
            self._speak("Invalid comment, either the word left or the word right must be mentioned in the voice recording.")


    def _speak(self, text):
        language = 'en'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("messageOfComputer.mp3")
        os.system("start messageOfComputer.mp3")
