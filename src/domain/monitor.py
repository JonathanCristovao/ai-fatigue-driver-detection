class DriverMonitor:
    def __init__(self, initial_speed=80, on_leds_update=None):
        self.speed = initial_speed
        self.max_speed = 110
        self.min_speed = 0
        self.deceleration_rate = 1.5
        self.acceleration_rate = 2.0
        self._on_leds_update = on_leds_update

    def set_leds_callback(self, callback):
        self._on_leds_update = callback

    def update(self, eyes_status, yawn_status, model_prediction):
        is_eyes_closed = 'closed' in eyes_status.lower()
        is_yawning = 'yawning' in yawn_status.lower()

        if is_eyes_closed:
            self.speed = max(self.min_speed, self.speed - self.deceleration_rate)
        else:
            self.speed = min(self.max_speed, self.speed + self.acceleration_rate)

        if self._on_leds_update:
            self._on_leds_update(eyes_closed=is_eyes_closed, yawning_detected=is_yawning)

        return self.speed, is_yawning
