import threading
import time

GPIO_AVAILABLE = False
try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
except (RuntimeError, ImportError):
    pass

if GPIO_AVAILABLE:
    import os
    os.environ['GPIOZERO_PIN_FACTORY'] = 'lgpio'

PINO_RED = 12
PINO_BLINK = 16
BLINK_INTERVAL = 0.5


class GPIOController:
    def __init__(self):
        self.led_vermelho = None
        self.led_amarelo = None
        self._eyes_closed = False
        self._yawning_active = False
        self._blink_thread = None
        self._running = True

        if GPIO_AVAILABLE:
            self._setup()

    def _setup(self):
        try:
            self.led_vermelho = LED(PINO_RED)
            self.led_amarelo = LED(PINO_BLINK)
            print("GPIO (gpiozero + lgpio) configurado com sucesso")
        except Exception as e:
            print(f"Erro ao configurar GPIO: {e}")

    def _led_blink_worker(self):
        while self._yawning_active and self._running:
            try:
                self.led_amarelo.on()
                time.sleep(BLINK_INTERVAL)
                self.led_amarelo.off()
                time.sleep(BLINK_INTERVAL)
            except:
                pass

    def update(self, eyes_closed, yawning_detected):
        if not GPIO_AVAILABLE or self.led_vermelho is None or self.led_amarelo is None:
            return

        if eyes_closed and not self._eyes_closed:
            self._eyes_closed = True
            try:
                self.led_vermelho.on()
            except:
                pass
            print("Alerta Fisico: Olhos Fechados -> LED Vermelho LIGADO")
        elif not eyes_closed and self._eyes_closed:
            self._eyes_closed = False
            try:
                self.led_vermelho.off()
            except:
                pass
            print("Status Fisico: Olhos Abertos -> LED Vermelho DESLIGADO")

        if yawning_detected and not self._yawning_active:
            self._yawning_active = True
            if self._blink_thread is None or not self._blink_thread.is_alive():
                self._blink_thread = threading.Thread(target=self._led_blink_worker, daemon=True)
                self._blink_thread.start()
            print("Alerta Fisico: Bocejo Detectado! -> LED Amarelo PISCANDO")
        elif not yawning_detected and self._yawning_active:
            self._yawning_active = False
            try:
                self.led_amarelo.off()
            except:
                pass
            print("Status Fisico: Bocejo Encerrado -> LED Amarelo DESLIGADO")

    def cleanup(self):
        self._running = False
        if GPIO_AVAILABLE:
            if self.led_vermelho is not None:
                self.led_vermelho.off()
            if self.led_amarelo is not None:
                self.led_amarelo.off()

    @property
    def is_available(self):
        return GPIO_AVAILABLE
