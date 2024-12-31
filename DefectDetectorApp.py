import pyaudio
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import sys
import librosa
import math

# Parametry zvuku
CHUNK = 1024  # Velikost bufferu
FORMAT = pyaudio.paInt16  # Formát zvuku
CHANNELS = 1  # Mono
RATE = 44100  # Vzorkovací frekvence
MAX_TIME = 8  # Maximální doba záznamu zobrazená v grafu (v sekundách)

# Parametry
D = 1 # Průměr kola (m)
v = 70 / 3.6  # Rychlost vlaku (m/s)
T = 1 / RATE  # Perioda vzorkování
Gamma = int(math.ceil(np.pi * D / (v * T)))  # Velikost bloku

# RMS
W = 500  # Velikost okna pro RMS
F = 20   # offset prekrytí okna
M = math.floor((Gamma-W)/F + 1) #pocet rms hodnot v jednom bloku b

# Nastavení prahových hodnot
alpha = 1100  # Maximální časová odchylka
beta = 1.7  # Maximální logaritmická odchylka výkonu
zeta = 0.7  # Nastavená prahová hodnota pro detekci defektu 

# Inicializace záznamu
audio = pyaudio.PyAudio()
stream = None
is_recording = False
time_d = 0.0

# Uchování signálu
recorded_signal = np.zeros(RATE * MAX_TIME)

"""
Nacteni signalu ze souboru mp3,
zpracovini signalu, detekce vady
"""
def load_file():
    global RATE, recorded_signal, t
    recorded_signal = np.zeros(RATE * MAX_TIME)
    filename = "strecno.mp3"
    signal, RATE = librosa.load(filename, sr=None)  # sr=None zachová původní vzorkovací frekvenci
    t = time.time()

    process_file_audio(signal)

"""
Zpracovani signalu, vykresleni zpracovavaneho signalu, detekce vady
input: signal 
"""
def process_file_audio(signal):
    global recorded_signal, v, Gamma, W, F, M, time_d, t
    total_length = len(signal)
    segment_length = RATE * 3  # Process 3 seconds of audio
    step_size = RATE * 1      # Shift by 2 seconds


    for start in range(0, total_length, step_size):
        time_sec.config(text="Čas: " + str(start/RATE) + "s")
        time_d = start/RATE
        #print("Processing segment", start/RATE, "to", (start + segment_length)//RATE)
        # Extract the current segment
        end = start + segment_length
        block = signal[start:end]
        if len(block) < segment_length:
            # Zero-pad the block if it's shorter than the segment length
            block = np.pad(block, (0, segment_length - len(block)))

        # Update the rolling buffer with the new block
        recorded_signal = np.roll(recorded_signal, -len(block))
        recorded_signal[-len(block):] = block

        # Update the plot
        ax.clear()
        ax.plot(np.linspace(-MAX_TIME, 0, len(recorded_signal)), recorded_signal)
        ax.set_ylim([-1, 1])  # Normalized signal range
        ax.set_xlim([-MAX_TIME, 0])
        ax.set_xlabel("Čas (s)", fontsize=20)
        ax.set_ylabel("Amplituda", fontsize=20)
        ax.set_title("Zvukový signál", fontsize=20)
        plt.tick_params(axis='both', labelsize=18)
        canvas.draw()
        canvas.flush_events()  # Ensure the GUI updates immediately

        # # Process the current block
        process_audio(block)

        # # Wait 2 seconds before processing the next segment
        root.update()  # Allow GUI updates    
        #time.sleep(1)
        if 1-(time.time()-t) > 0:
            time.sleep(1-(time.time()-t))
        t = time.time()


"""
vypocet M rms hodnot, pomoci klouzaceho okna velikosti W s offsetem F 
input:  signal - signal, ze ktereho se rms hodnoty pocitaji
        M - vysledny pocet rms hodnot
        W - velikost klouzaveho okna
        F - offset posunu klouzaveho okna
output: rms hodnoty
"""
def calculate_rms(signal, M, W, F):
    rms =  []
    for i in range(0, M):
        rms.append(np.sqrt((1/W)*np.sum(np.abs(signal[i*F:i*F+W])**2)))
    return rms


"""
vypocet celkoveho skore signalu
input:  time_deviations - casove odchylky mezi spickami signalu
        power_deviations - vykonove odchylky mezi spickami signalu
        alpha - prahova hodnota pro casove odchylky
        beta - prahova hodnota pro vykonove odchylky 
        B - pocet bloku signalu
output: skore signalu
"""
def defect_score_curve(time_deviations, power_deviations, alpha, beta, B):
    omega = []
    time = []
    power = []
    for td, pd in zip(time_deviations, power_deviations):
        if int(abs(td - Gamma) <= alpha and abs(np.log10(pd)) <= beta):
            omega.append(1)
        else:
            omega.append(0)
        time.append(abs(td - Gamma))
        power.append(abs(np.log10(pd)))

    if np.median(time) == 0:
        return 1e-10

    return sum(omega) / (B-1)

"""
nalezeni spicek signalu, casove a vykonove odchylky mezi spickami 
input:  signal - signal, ve kterem se spicky hledaji
        gamma - vellikost bloku signalu
        W - velikost klouzaveho okna pro vypocet rms
        F - offdet posunu klouzaveho okna pro vypocet rms
output: peaks - spicky signalu
        time_deviations - casove odchylky spicek
        power_deviations - vykonove odchylky spicek    
"""
def signal_peaks(signal, Gamma, W, F):
    blocks = []
    for i in range(0, len(signal), Gamma):
        if len(signal[i:i + Gamma]) == Gamma:
            blocks.append(signal[i:i + Gamma])

    # Detekce vrcholů v každém bloku
    peaks = []
    for count, block in enumerate(blocks):
        rms = calculate_rms(block, M, W, F)  # Nastavte vhodnou velikost okna
        rms_peak_idx = np.argmax(rms)
        p = np.argmax(abs(signal[rms_peak_idx*F + count*Gamma:rms_peak_idx*F + W-1 + count*Gamma]))
        if signal[rms_peak_idx*F + p + count*Gamma]== 0:
            peaks.append((rms_peak_idx*F + p + count*Gamma, 1e-10))
        else: 
            peaks.append((rms_peak_idx*F + p + count*Gamma, signal[rms_peak_idx*F + p + count*Gamma]))
        
    peaks = np.array(peaks)

    time_deviations = [peaks[i + 1][0] - peaks[i][0] for i in range(len(peaks) - 1)]  # Časové odchylky mezi špičkami
    power_deviations = [abs(peaks[i + 1][1]) / abs(peaks[i][1]) for i in range(len(peaks) - 1)]  # Odchylka výkonu mezi špičkami 

    return peaks, time_deviations, power_deviations
    

""""
vyhodnoceni signalu, zda obsauje zaznam vady ci nikoli
input:  time_deviations - casove odchylky spicek
        power_deviations - vykonove odchylky spicek 
        alpha - prahova hodnota pro casove odchylky
        beta - prahova hodnota pro vykonove odchylky 
        zeta - prahova hodnota pro skore signalu
        B - pocet bloku signalu
"""
def signal_eval(time_deviations, power_deviations, alpha, beta, zeta, B):
    defect_score = defect_score_curve(time_deviations, power_deviations, alpha, beta, B)

    if defect_score > zeta and np.mean(time_deviations) != 0:
        update_text("Detekován defekt kola!", defect_score, "red")
    else:
        update_text("Signál je normální.", defect_score, "green")



"""
Zpracovani a vyhodnoceni signalu
input:  data - signal k zpravovani a vyhodnoceni
"""
def process_audio(data):
    B = math.floor(len(data)/Gamma) # pocet bloku v signalu

    if B > 1:
        peaks, time_der, power_der = signal_peaks(data, Gamma, W, F)
        signal_eval(time_der, power_der, alpha, beta, zeta, B)

"""
spusteni nahravani audia
"""
def start_recording():
    global stream, is_recording, recorded_signal
    recorded_signal = np.zeros(RATE * MAX_TIME)
    if not is_recording:
        is_recording = True
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        threading.Thread(target=record_audio).start()

"""
zastaveni nahravani audia
"""
def stop_recording():
    global stream, is_recording
    if is_recording:
        is_recording = False
        if stream is not None:
            stream.stop_stream()
            stream.close()
            stream = None

"""
ukonceni programu
"""
def exit_program():
    stop_recording()
    audio.terminate()
    root.destroy()
    sys.exit()

"""
Nahravani zvuku z mikrofonu a zpracovani
"""
def record_audio():
    global is_recording, recorded_signal
    frames = []
    start_time = time.time()

    while is_recording:
        data = stream.read(CHUNK)
        signal = np.frombuffer(data, dtype=np.int16)

        # Posun a aktualizace zaznamenaného signálu
        recorded_signal = np.roll(recorded_signal, -len(signal))
        recorded_signal[-len(signal):] = signal
        frames.append(signal)

        # Aktualizace grafu
        ax.clear()
        ax.plot(np.linspace(-MAX_TIME, 0, len(recorded_signal)), recorded_signal)
        ax.set_ylim([-32768, 32767])  # Rozsah 16bitového zvuku
        ax.set_xlim([-MAX_TIME, 0])  # Časový rozsah v sekundách
        ax.set_xlabel("Čas (s)")
        ax.set_ylabel("Amplituda")
        ax.set_title("Zvukový signál")
        canvas.draw()

        # Každé 3 sekundy zavolat zpracování
        if time.time() - start_time >= 2:
            time_sec.config(text="Čas: " + str(time.time) + "s")
            process_audio(np.concatenate(frames))
            frames = frames[len(frames) // 2:]
            start_time = time.time()

"""
aktualizace textu
"""
def update_text(text, score, color):
    vysledek_text.config(text=text, foreground=color)
    vysledek_cislo.config(text=f"Skóre: {round(score, 2)}")
    
 
# Vytvoření GUI
root = tk.Tk()
root.title("Zvukový záznam")
root.geometry("800x800")  # Čtvercové okno

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Tlačítka nahoře
button_frame = ttk.Frame(main_frame)
button_frame.pack(side=tk.TOP, pady=10)

s = ttk.Style()
s.configure('my.TButton', font=('Arial', 18))

start_button = ttk.Button(button_frame, text="Start", command=start_recording, style='my.TButton')
start_button.grid(column=0, row=0, padx=10)
start_button.config(width=20)

stop_button = ttk.Button(button_frame, text="Stop", command=stop_recording, style='my.TButton')
stop_button.grid(column=1, row=0, padx=10)
stop_button.config(width=20)

load_file_button = ttk.Button(button_frame, text="Nahrát z MP3", command=load_file, style='my.TButton')
load_file_button.grid(column=3, row=0, padx=10)
load_file_button.config(width=20)

exit_button = ttk.Button(button_frame, text="Ukončit", command=exit_program, style='my.TButton')
exit_button.grid(column=2, row=0, padx=10)
exit_button.config(width=20)


# Graf uprostřed
fig, ax = plt.subplots(figsize=(7, 7))  # Velký čtvercový graf
ax.set_ylim([-32768, 32767])
ax.set_xlim([-MAX_TIME, 0])
ax.set_xlabel("Čas (s)", fontsize=20)
ax.set_ylabel("Amplituda", fontsize=20)
ax.set_title("Zvukový signál", fontsize=20)
plt.tick_params(axis='both', labelsize=18)
canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

# Text dole
vysledek_cislo = ttk.Label(main_frame, text="Číslo signálu se zobrazí zde.", anchor="center", font=("Arial", 20))
vysledek_cislo.pack(side=tk.BOTTOM, pady=8)

time_sec = ttk.Label(main_frame, text="time se zobrazí zde.", anchor="w", font=("Arial", 20))
time_sec.pack(side=tk.BOTTOM, pady=8)

vysledek_text = ttk.Label(main_frame, text="Výsledek signálu se zobrazí zde.", anchor="center", font=("Arial", 28, "bold"))
vysledek_text.pack(side=tk.BOTTOM, pady=15)

# Ukončení aplikace
root.protocol("WM_DELETE_WINDOW", exit_program)
root.mainloop()
