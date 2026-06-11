import cv2
import src.config as conf
from src.domain.iris import Iris


class Eye:
    def __init__(self, frame, face_landmarks, id):
        self.frame = frame
        self.face_landmarks = face_landmarks
        self.id = id
        self.iris = Iris(frame, face_landmarks, id)
        self.pos = self._get_eye_pos()
        self.iris_relative_to_eye = self._get_gaze_ratio()
        self.eye_veti_to_hori = self._get_blink_ratio()

    def _get_eye_pos(self):
        h, w = self.frame.shape[:2]
        eye_pos = list()
        for id in self.id[:4]:
            pos = self.face_landmarks.landmark[id]
            cx = int(pos.x * w)
            cy = int(pos.y * h)
            eye_pos.append([cx, cy])
        return eye_pos

    def _get_gaze_ratio(self):
        ratiol = (self.pos[0][0] - self.iris.pos[1][0]) / (self.pos[0][0] - self.pos[1][0])
        ratioc = (self.pos[0][0] - self.iris.pos[0][0]) / (self.pos[0][0] - self.pos[1][0])
        ratior = (self.pos[0][0] - self.iris.pos[3][0]) / (self.pos[0][0] - self.pos[1][0])
        return [ratiol, ratioc, ratior]

    def gaze_left(self, threshold=conf.GAZE_LEFT):
        return self.iris_relative_to_eye[0] < threshold

    def gaze_right(self, threshold=conf.GAZE_RIGHT):
        return self.iris_relative_to_eye[2] > threshold

    def gaze_center(self):
        return not self.gaze_left() and not self.gaze_right()

    def _get_blink_ratio(self):
        return (self.pos[3][1] - self.pos[2][1]) / (self.pos[0][0] - self.pos[1][0])

    def eye_closed(self, threshold=conf.EYE_CLOSED):
        return self.eye_veti_to_hori < threshold

    def draw_eye(self):
        for pos in self.pos:
            cv2.circle(self.frame, pos, 2, conf.LM_COLOR, -1, lineType=cv2.LINE_AA)
