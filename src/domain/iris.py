import cv2
import src.config as conf


class Iris:
    def __init__(self, frame, face_landmarks, id):
        self.frame = frame
        self.face_landmarks = face_landmarks
        self.id = id
        self.pos = self._get_iris_pos()

    def _get_iris_pos(self):
        h, w = self.frame.shape[:2]
        iris_pos = list()
        for id in self.id[-5:]:
            pos = self.face_landmarks.landmark[id]
            cx = int(pos.x * w)
            cy = int(pos.y * h)
            iris_pos.append((cx, cy))
        return iris_pos

    def draw_iris(self, border=False):
        cv2.circle(self.frame, self.pos[0], 2, conf.LM_COLOR, -1, lineType=cv2.LINE_AA)
        if border:
            for pos in self.pos[1:]:
                cv2.circle(self.frame, pos, 1, conf.LM_COLOR, -1, lineType=cv2.LINE_AA)
