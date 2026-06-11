import cv2
import mediapipe as mp
import numpy as np
import time
import facial_tracking.conf as conf
import os


class _LandmarkList:
    def __init__(self, landmarks):
        self._landmarks = landmarks

    @property
    def landmark(self):
        return self._landmarks

    def __len__(self):
        return len(self._landmarks)


class _MeshResult:
    def __init__(self, landmarks):
        self.multi_face_landmarks = landmarks


class FaceMesh:
    def __init__(self, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.frame = None
        self.mesh_result = _MeshResult([])

        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo face_landmarker.task não encontrado em {model_path}. "
                "Baixe de: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        self.frame = frame
        self._face_mesh()

    def _face_mesh(self):
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if result.face_landmarks:
            wrapped = [_LandmarkList(face) for face in result.face_landmarks]
            self.mesh_result = _MeshResult(wrapped)
        else:
            self.mesh_result = _MeshResult([])

    def draw_mesh(self):
        pass

    def draw_mesh_eyes(self):
        if not self.mesh_result.multi_face_landmarks:
            return
        for face_landmarks in self.mesh_result.multi_face_landmarks:
            for idx in conf.LEFT_EYE[:4] + conf.RIGHT_EYE[:4]:
                h, w = self.frame.shape[:2]
                lm = face_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(self.frame, (cx, cy), 2, conf.CT_COLOR, -1, lineType=cv2.LINE_AA)

    def draw_mesh_lips(self):
        if not self.mesh_result.multi_face_landmarks:
            return
        for face_landmarks in self.mesh_result.multi_face_landmarks:
            for idx in conf.LIPS:
                h, w = self.frame.shape[:2]
                lm = face_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(self.frame, (cx, cy), 2, conf.CT_COLOR, -1, lineType=cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    fm = FaceMesh()
    ptime = 0
    ctime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        fm.process_frame(frame)
        fm.draw_mesh()

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30,30), 0, 0.8,
                    conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Face Mesh', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
