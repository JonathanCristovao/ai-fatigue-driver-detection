import cv2
import numpy as np
import math


class SpeedometerDrawer:
    def __init__(self, width=450, height=450):
        self.width = width
        self.height = height
        self.center = (width // 2, height // 2)
        self.radius = 160
        self.max_speed = 120
        self.start_angle = 135
        self.end_angle = 405
        self.total_angle = self.end_angle - self.start_angle

    def draw_speedometer(self, speed, is_warning=False):
        speedometer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(speedometer, (0, 0), (self.width, self.height), (24, 24, 24), -1)

        cv2.ellipse(speedometer, self.center, (self.radius, self.radius), 0,
                    self.start_angle, self.end_angle, (45, 45, 45), 6, cv2.LINE_AA)

        speed_normalized = min(max(speed, 0), self.max_speed)

        if is_warning:
            color_accent = (0, 0, 255)
        elif speed_normalized > 80:
            color_accent = (0, 165, 255)
        else:
            color_accent = (0, 200, 100)

        current_angle = self.start_angle + (speed_normalized / self.max_speed) * self.total_angle
        if speed_normalized > 0:
            cv2.ellipse(speedometer, self.center, (self.radius, self.radius), 0,
                        self.start_angle, current_angle, color_accent, 8, cv2.LINE_AA)

        for i in range(0, self.max_speed + 1, 20):
            angle = self.start_angle + (i / self.max_speed) * self.total_angle
            rad = math.radians(angle)

            x1 = int(self.center[0] + (self.radius - 12) * math.cos(rad))
            y1 = int(self.center[1] + (self.radius - 12) * math.sin(rad))
            x2 = int(self.center[0] + (self.radius - 2) * math.cos(rad))
            y2 = int(self.center[1] + (self.radius - 2) * math.sin(rad))
            cv2.line(speedometer, (x1, y1), (x2, y2), (180, 180, 180), 2, cv2.LINE_AA)

            x_text = int(self.center[0] + (self.radius - 32) * math.cos(rad))
            y_text = int(self.center[1] + (self.radius - 32) * math.sin(rad))
            offset_x = 10 if i >= 100 else (8 if i >= 10 else 4)
            cv2.putText(
                speedometer, str(i), (x_text - offset_x, y_text + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA
            )

        rad_needle = math.radians(current_angle)
        x_needle = int(self.center[0] + (self.radius - 15) * math.cos(rad_needle))
        y_needle = int(self.center[1] + (self.radius - 15) * math.sin(rad_needle))
        cv2.line(speedometer, self.center, (x_needle, y_needle), color_accent, 3, cv2.LINE_AA)

        cv2.circle(speedometer, self.center, 14, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.circle(speedometer, self.center, 8, color_accent, -1, cv2.LINE_AA)
        cv2.circle(speedometer, self.center, 3, (255, 255, 255), -1, cv2.LINE_AA)

        speed_text = f"{int(speed)}"
        cv2.putText(
            speedometer, speed_text, (self.center[0] - 35, self.center[1] + 65),
            cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            speedometer, "km/h", (self.center[0] - 20, self.center[1] + 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA
        )

        return speedometer
