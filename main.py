import os
import time
import cv2
import src.config as conf
from src.application.tflite_tracker import FacialTrackerTFLite
from src.domain.monitor import DriverMonitor
from src.presentation.speedometer import SpeedometerDrawer

if 'DISPLAY' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    from src.infrastructure.gpio_controller import GPIOController
    gpio = GPIOController()
    has_gpio = gpio.is_available
except Exception:
    gpio = None
    has_gpio = False


def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)

    try:
        facial_tracker_tflite = FacialTrackerTFLite(tflite_model_path='models/model_random.tflite')
    except Exception as e:
        print(f"Nao foi possivel carregar o modelo de rede neural: {e}")
        cap.release()
        return

    speedometer_drawer = SpeedometerDrawer(width=450, height=450)
    driver_monitor = DriverMonitor(initial_speed=80)

    if has_gpio:
        driver_monitor.set_leds_callback(gpio.update)

    ptime = 0
    has_display = 'DISPLAY' in os.environ

    print("\n[INFO] Monitoramento Ativo.")
    print("[INFO] Pressione 'q' na janela ou 'Ctrl+C' no terminal para finalizar\n")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            facial_tracker_tflite.process_frame(frame)

            ctime = time.time()
            time_delta = ctime - ptime
            fps = 1 / time_delta if time_delta > 0.000001 else 0
            ptime = ctime

            frame = cv2.flip(frame, 1)

            scale_factor = 1.5
            new_w = int(frame.shape[1] * scale_factor)
            new_h = int(frame.shape[0] * scale_factor)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            eyes_status = facial_tracker_tflite.facial_tracker.eyes_status if facial_tracker_tflite.facial_tracker.detected else ""
            yawn_status = facial_tracker_tflite.facial_tracker.yawn_status if facial_tracker_tflite.facial_tracker.detected else ""
            model_prediction = facial_tracker_tflite.prediction

            current_speed, show_alert = driver_monitor.update(eyes_status, yawn_status, model_prediction)

            cv2.putText(frame, f'FPS: {int(fps)}', (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)

            if facial_tracker_tflite.facial_tracker.detected:
                y_offset = 100
                cv2.putText(frame, f'{eyes_status}', (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.1, conf.WARN_COLOR, 3, lineType=cv2.LINE_AA)
                y_offset += 55
                cv2.putText(frame, f'{yawn_status}', (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.1, conf.WARN_COLOR, 3, lineType=cv2.LINE_AA)
                y_offset += 55

                if model_prediction:
                    prediction_text = f'{model_prediction}: {facial_tracker_tflite.confidence:.2%}'
                    color = conf.WARN_COLOR if 'Fatigue' in model_prediction else (0, 255, 0)
                    cv2.putText(frame, prediction_text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, lineType=cv2.LINE_AA)

            if has_display:
                cv2.imshow('Facial Tracking com TFLite', frame)

            speedometer = speedometer_drawer.draw_speedometer(current_speed, show_alert)

            if show_alert:
                h, w = speedometer.shape[:2]
                cv2.rectangle(speedometer, (0, 0), (w, 45), (0, 0, 200), -1)
                cv2.putText(speedometer, "ATTENTION!", (w // 2 - 65, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            status_text = f"Speed: {int(current_speed)} km/h"
            cv2.putText(speedometer, status_text, (20, speedometer.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)

            if has_display:
                cv2.imshow('Velocimetro - Driver Monitor', speedometer)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupcao capturada.")
    finally:
        cap.release()
        if has_display:
            cv2.destroyAllWindows()
        if has_gpio:
            gpio.cleanup()
        print("[INFO] Execucao finalizada.")


if __name__ == '__main__':
    main()
