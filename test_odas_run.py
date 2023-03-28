import subprocess
import os
import keyboard

# Change directory to ~/odas/bin
os.chdir('/home/pi/odas/bin')

# Start matrix-odas
matrix_odas_process = subprocess.Popen(["./matrix-odas"])

# Start odaslive with the given configuration file
odaslive_process = subprocess.Popen(["./odaslive", "-vc", "../config/matrix-demo/matrix_voice.cfg"])

while True:
    if keyboard.is_pressed('q'):
        matrix_odas_process.terminate()
        odaslive_process.terminate()
        break