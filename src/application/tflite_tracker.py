import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

from src.application.facial_tracker import FacialTracker


class FacialTrackerTFLite:
    def __init__(self, tflite_model_path='models/model_random.tflite'):
        self.facial_tracker = FacialTracker()
        self.tflite_model_path = tflite_model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = ['Alertness', 'Fatigue']
        self.prediction = None
        self.confidence = None
        self._load_tflite_model()

    def _load_tflite_model(self):
        try:
            self.interpreter = tflite.Interpreter(model_path=self.tflite_model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Modelo TFLite carregado: {self.tflite_model_path}")
        except Exception as e:
            print(f"Erro ao carregar o modelo TFLite: {e}")
            raise

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame.astype(np.float32) / 127.5 - 1.0
        return normalized

    def predict_fatigue(self, frame):
        try:
            input_data = self.preprocess_frame(frame)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data.astype(self.input_details[0]['dtype'])

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions = output_data[0]
            predicted_class = np.argmax(predictions)

            self.prediction = self.class_names[predicted_class]
            self.confidence = predictions[predicted_class]
            return self.prediction, self.confidence
        except Exception as e:
            print(f"Erro na inferencia do modelo: {e}")
            return None, None

    def process_frame(self, frame):
        self.facial_tracker.process_frame(frame)
        if self.facial_tracker.detected:
            self.predict_fatigue(frame)
        else:
            self.prediction = None
            self.confidence = None
