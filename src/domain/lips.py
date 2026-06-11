import cv2
import src.config as conf


class Lips:
    def __init__(self, frame, face_landmarks, id):
        self.frame = frame
        self.face_landmarks = face_landmarks
        self.id = id
        self.pos = self._get_lips_pos()
        self.mouth_open_ratio = self._get_open_ratio()

    def _get_lips_pos(self):
        h, w = self.frame.shape[:2]
        lips_pos = list()
        for id in self.id:
            pos = self.face_landmarks.landmark[id]
            cx = int(pos.x * w)
            cy = int(pos.y * h)
            lips_pos.append([cx, cy])
        return lips_pos

    def _get_open_ratio(self):
        return (self.pos[3][1] - self.pos[2][1]) / (self.pos[0][0] - self.pos[1][0])

    def mouth_open(self, threshold=conf.MOUTH_OPEN):
        return self.mouth_open_ratio > threshold

    def draw_lips(self):
        for pos in self.pos:
            cv2.circle(self.frame, pos, 1, conf.LM_COLOR, -1, lineType=cv2.LINE_AA)
