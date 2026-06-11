from src.infrastructure.face_mesh import FaceMesh
from src.domain.eye import Eye
from src.domain.lips import Lips
import src.config as conf


class FacialTracker:
    def __init__(self):
        self.fm = FaceMesh()
        self.left_eye = None
        self.right_eye = None
        self.lips = None
        self.left_eye_closed_frames = 0
        self.right_eye_closed_frames = 0
        self.eyes_status = ''
        self.yawn_status = ''
        self.detected = False

    def process_frame(self, frame):
        self.detected = False
        self.fm.process_frame(frame)
        self.fm.draw_mesh_lips()

        if self.fm.mesh_result.multi_face_landmarks:
            self.detected = True
            for face_idx, face_landmarks in enumerate(self.fm.mesh_result.multi_face_landmarks):
                try:
                    self.left_eye = Eye(frame, face_landmarks, conf.LEFT_EYE)
                    self.right_eye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
                    self.lips = Lips(frame, face_landmarks, conf.LIPS)
                    self._check_eyes_status()
                    self._check_yawn_status()
                except Exception as e:
                    print(f"[ERROR] Erro ao processar face {face_idx}: {e}")

    def _check_eyes_status(self):
        self.eyes_status = ''

        if self.left_eye.eye_closed():
            self.left_eye_closed_frames += 1
        else:
            self.left_eye_closed_frames = 0
            self.left_eye.iris.draw_iris(True)

        if self.right_eye.eye_closed():
            self.right_eye_closed_frames += 1
        else:
            self.right_eye_closed_frames = 0
            self.right_eye.iris.draw_iris(True)

        if self._left_eye_closed() or self._right_eye_closed():
            self.eyes_status = 'eye closed'
            return

        if not self.left_eye.eye_closed() and not self.right_eye.eye_closed():
            if self.left_eye.gaze_right() and self.right_eye.gaze_right():
                self.eyes_status = 'gazing right'
            elif self.left_eye.gaze_left() and self.right_eye.gaze_left():
                self.eyes_status = 'gazing left'
            elif self.left_eye.gaze_center() and self.right_eye.gaze_center():
                self.eyes_status = 'gazing center'

    def _check_yawn_status(self):
        self.yawn_status = ''
        if self.lips.mouth_open():
            self.yawn_status = 'yawning'

    def _left_eye_closed(self, threshold=conf.FRAME_CLOSED):
        return self.left_eye_closed_frames > threshold

    def _right_eye_closed(self, threshold=conf.FRAME_CLOSED):
        return self.right_eye_closed_frames > threshold
