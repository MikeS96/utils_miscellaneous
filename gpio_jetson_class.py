# coding: utf-8

import RPi.GPIO as GPIO
import time
from threading import Thread

GPIO.setwarnings(False)

channels = [18, 17]
GPIO.setmode(GPIO.BCM)
GPIO.setup(channels, GPIO.OUT, initial=GPIO.HIGH)


class GpioSystem:
    def __init__(self, ts=0.2):

        self.tsignal = ts

        self.out_1 = 18
        self.out_2 = 17

        self.mode_signal_1 = True
        self.mode_signal_2 = True

    def turn_on_gpio(self, cout):

        if cout == 18 and self.mode_signal_1:
            self.mode_signal_1 = False
            t = Thread(target=self.gpio_thread, name='Rely_1', args=(cout,))
            t.daemon = True
            t.start()
            return self
        elif cout == 17 and self.mode_signal_2:
            self.mode_signal_2 = False
            t = Thread(target=self.gpio_thread, name='Rely_2', args=(cout,))
            t.daemon = True
            t.start()
            return self
        return self

    def gpio_thread(self, out_pin):

        curr_value = GPIO.LOW
        GPIO.output(out_pin, curr_value)
        time.sleep(self.tsignal)
        curr_value ^= GPIO.HIGH
        GPIO.output(out_pin, curr_value)
        if out_pin == 18:
            self.mode_signal_1 = True
        elif out_pin == 17:
            self.mode_signal_2 = True

    def set_output_low(self):

        curr_value = GPIO.HIGH
        GPIO.output(self.out_1, curr_value)
        GPIO.output(self.out_2, curr_value)

    def gpio_cleanup(self):

        curr_value = GPIO.HIGH
        GPIO.output(self.out_1, curr_value)
        GPIO.output(self.out_2, curr_value)
        GPIO.cleanup([self.out_1, self.out_2])

