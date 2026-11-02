import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import pendulum_tracker 
import matplotlib.pyplot as plt
import psutil
import time
import csv
from datetime import datetime
import os
import threading

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.title("SISTEMA DE RASTREAMENTO - PENDULO DUPLO")
        self.geometry("900x600")

        self.tracker = pendulum_tracker.PendulumTracker()
        self.cap = cv2.VideoCapture(0)
        
        self.mode = "PREVIEW" 
        self.countdown_value = 3
        self.frame_count = 0

        self.history_x1, self.history_y1 = [], []
        self.history_x2, self.history_y2 = [], []

        self.start_time = 0
        self.time_history = []
        self.thread_time_history = []

        self.calib_orig = None
        self.calib_align = None

        MAX_THREADS = 8   
        self.process = psutil.Process(os.getpid())
        self.main_tid = threading.get_native_id()
        self.thread_ids = [self.main_tid]

        self.thread_usage = {}
        self.previous_thread_times = {}
        self.previous_sample_time = time.perf_counter()
        
        self.num_threads = MAX_THREADS
        self.thread_history = [[] for _ in range(self.num_threads)]

        # --- LAYOUT ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.title_label = ctk.CTkLabel(self.sidebar, text="CONTROLES", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        self.btn_play = ctk.CTkButton(self.sidebar, text="PLAY", fg_color="green", command=self.start_video)
        self.btn_play.pack(padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="STOP", fg_color="red", command=self.stop_video)
        self.btn_stop.pack(padx=20, pady=10)

        self.btn_calib = ctk.CTkButton(self.sidebar, text="Calibrar Eixos", fg_color="orange", command=self.start_calibration)
        self.btn_calib.pack(padx=20, pady=10)

        self.btn_plot = ctk.CTkButton(self.sidebar, text="Gráfico do Pêndulo", fg_color="blue", command=self.plot_graph)
        self.btn_plot.pack(padx=20, pady=10)

        self.btn_plot_threads = ctk.CTkButton(self.sidebar, text="Gráfico de Threads", fg_color="purple", command=self.plot_threads)
        self.btn_plot_threads.pack(padx=20, pady=10)

        # --- PAINEL DE MONITORAMENTO DE THREADS ---
        self.lbl_threads = ctk.CTkLabel(self.sidebar, text="USO DE THREADS", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_threads.pack(padx=20, pady=(30, 5))

        self.thread_bars = []
        self.thread_labels = []
        
        for i in range(self.num_threads): 
            frame_bar = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            frame_bar.pack(fill="x", padx=10, pady=2)
            
            lbl = ctk.CTkLabel(frame_bar, text=f"T{i}: 0%", width=45, anchor="w")
            lbl.pack(side="left")
            
            bar = ctk.CTkProgressBar(frame_bar, width=100, progress_color="#00a86b")
            bar.set(0)
            bar.pack(side="right", padx=5)
            
            self.thread_labels.append(lbl)
            self.thread_bars.append(bar)

        self.video_container = ctk.CTkFrame(self)
        self.video_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(self.video_container, text="")
        self.video_label.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_system_stats()
        self.update_frame()
        self.video_label.bind("<Button-1>", self.on_video_click)

    def start_video(self):
        if self.mode == "PREVIEW":
            self.history_x1.clear(); self.history_y1.clear()
            self.history_x2.clear(); self.history_y2.clear()
            self.time_history.clear()
            self.thread_time_history.clear()
            for th in self.thread_history: th.clear()
            
            self.frame_count = 0
            self.mode = "COUNTDOWN"
            self.countdown_value = 3
            self.tick_countdown()

    def tick_countdown(self):
        if self.countdown_value > 1:
            self.countdown_value -= 1
            self.after(1000, self.tick_countdown)
        else:
            self.mode = "TRACKING"
            self.start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"../../CSV/pendulo/pendulo_log_{timestamp}.csv"
            # Inicia as threads no C++
            self.tracker.start_tracking(csv_filename)

    def stop_video(self):
        if self.mode == "TRACKING":
            self.tracker.stop_tracking()
            if len(self.time_history) > 0:
                self.save_threads_csv()
        self.mode = "PREVIEW"
        
    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.mode == "PREVIEW":
                    cv2.putText(frame, "PREVIEW - ALINHE O PENDULO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                    if self.calib_orig and self.calib_align:
                        cv2.line(frame, self.calib_orig, self.calib_align, (0, 255, 0), 2) # Linha do Eixo Y
                        cv2.circle(frame, self.calib_orig, 6, (0, 0, 255), -1)             # Ponto Origem (0,0)
                        cv2.putText(frame, "EIXOS CALIBRADOS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif self.mode == "CALIBRATION_STEP1":
                    cv2.putText(frame, "CLIQUE NO ROLAMENTO (ORIGEM)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                elif self.mode == "CALIBRATION_STEP2":
                    cv2.putText(frame, "CLIQUE NA HASTE INFERIOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.circle(frame, self.calib_orig, 5, (0, 255, 255), -1)
                
                elif self.mode == "COUNTDOWN":
                    cv2.putText(frame, str(self.countdown_value), (280, 280), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 165, 255), 10)
                    cv2.putText(frame, "PREPARANDO...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                elif self.mode == "TRACKING":
                    cv2.putText(frame, "GRAVANDO!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    elapsed = time.time() - self.start_time
                    self.tracker.submit_frame(frame, self.frame_count, elapsed)
                    self.frame_count += 1

                    results = self.tracker.get_latest_results()
                    box = 25
                    last_x1_raw, last_y1_raw, last_x2_raw, last_y2_raw = 0, 0, 0, 0
                    
                    for res in results:
                        f_id, elp, x1_raw, y1_raw, x2_raw, y2_raw, x1, y1, x2, y2 = res
                        self.time_history.append(elp)
                        
                        if x1_raw > 0 and y1_raw > 0:
                            self.history_x1.append(x1)
                            self.history_y1.append(y1) # Sem o sinal de '-' manual (o C++ já inverteu)
                            last_x1_raw, last_y1_raw = x1_raw, y1_raw
                        if x2_raw > 0 and y2_raw > 0:
                            self.history_x2.append(x2)
                            self.history_y2.append(y2)
                            last_x2_raw, last_y2_raw = x2_raw, y2_raw

                    # Desenha as miras usando OS PIXELS BRUTOS, não a matemática física
                    if last_x1_raw > 0 and last_y1_raw > 0:
                        cv2.rectangle(frame, (int(last_x1_raw)-box, int(last_y1_raw)-box), (int(last_x1_raw)+box, int(last_y1_raw)+box), (0, 0, 255), 2)
                        cv2.circle(frame, (int(last_x1_raw), int(last_y1_raw)), 4, (0, 255, 0), -1) 
                    if last_x2_raw > 0 and last_y2_raw > 0:
                        cv2.rectangle(frame, (int(last_x2_raw)-box, int(last_y2_raw)-box), (int(last_x2_raw)+box, int(last_y2_raw)+box), (255, 0, 0), 2)
                        cv2.circle(frame, (int(last_x2_raw), int(last_y2_raw)), 4, (0, 255, 0), -1)
                
                cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_img)
                img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk

        self.after(15, self.update_frame)

    def update_system_stats(self):
        # Mapeia os TIDs nativos diretamente do C++ para a GUI
        cpp_tids = self.tracker.get_thread_ids()
        workers_tids = cpp_tids.get("workers", [])
        writer_tid = cpp_tids.get("writer", 0)
        
        self.thread_ids = [self.main_tid] + workers_tids
        if writer_tid > 0:
            self.thread_ids.append(writer_tid)

        self.collect_thread_usage()
    
        # Registra o tempo exato desta medição das threads
        if self.mode == "TRACKING":
            elapsed = time.time() - self.start_time
            self.thread_time_history.append(elapsed)

        for i, tid in enumerate(self.thread_ids):
            if i >= self.num_threads:
                break
                
            usage = self.thread_usage.get(tid, 0.0)
            
            role = " (GUI)" if i == 0 else " (WRITER)" if i == self.num_threads - 1 else f" (W{i})"
            self.thread_labels[i].configure(text=f"T{i}{role}: {usage:.1f}%")
    
            color = "#ff3333" if usage > 70 else "#00a86b"
            self.thread_bars[i].configure(progress_color=color)
            self.thread_bars[i].set(usage / 100.0)
    
            if self.mode == "TRACKING":
                self.thread_history[i].append(usage)
    
        self.after(500, self.update_system_stats)

    def collect_thread_usage(self):
        now = time.perf_counter()
        elapsed = now - self.previous_sample_time
        if elapsed <= 0: return

        current = {thread.id: thread.user_time + thread.system_time for thread in self.process.threads()}
        usage = {}
        for tid, cpu_time in current.items():
            previous = self.previous_thread_times.get(tid)
            if previous is not None:
                delta_cpu = max(0.0, cpu_time - previous)
                usage[tid] = min(100.0, 100.0 * delta_cpu / elapsed)

        self.previous_thread_times = current
        self.previous_sample_time = now
        self.thread_usage = usage

    def plot_graph(self):
        if not self.history_x1 and not self.history_x2: return 
        plt.style.use('dark_background')
        plt.figure(figsize=(8, 6))
        if self.history_x1: plt.plot(self.history_x1, self.history_y1, color='red', label='Massa 1')
        if self.history_x2: plt.plot(self.history_x2, self.history_y2, color='blue', label='Massa 2')
        plt.title('TRAJETORIA DO PENDULO DUPLO')
        plt.xlabel('Posição X'); plt.ylabel('Posição Y')
        plt.legend(); plt.grid(True, alpha=0.3); plt.show()

    def plot_threads(self):
        if not self.thread_time_history: return 
        plt.style.use('dark_background')
        plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap("tab10")
        for i in range(self.num_threads):
            if self.thread_history[i]:
                limit = min(len(self.thread_time_history), len(self.thread_history[i]))
                plt.plot(self.thread_time_history[:limit], self.thread_history[i][:limit], color=cmap(i), label=f'Thread {i}')
        plt.title('GRAFICO DE USO DAS THREADS NO TEMPO')
        plt.xlabel('Tempo (s)'); plt.ylabel('Uso da CPU (%)')
        plt.legend(); plt.grid(True, alpha=0.3); plt.show()

    def save_threads_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../../CSV/threads/threads_log_{timestamp}.csv"
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(["Tempo (s)"] + [f"Thread {i} (%)" for i in range(self.num_threads)])
                for i in range(len(self.thread_time_history)):
                    t = f"{self.thread_time_history[i]:.2f}".replace('.', ',')
                    row = [t]
                    for k in range(self.num_threads):
                        row.append(self.thread_history[k][i] if i < len(self.thread_history[k]) else 0)
                    writer.writerow(row)
            print(f"Sucesso: Arquivo de threads salvo em {filename}")
        except Exception as e:
            print(f"Erro ao salvar CSV: {e}")

    def start_calibration(self):
        if self.mode == "PREVIEW":
            self.mode = "CALIBRATION_STEP1"

    def on_video_click(self, event):
        if self.mode == "CALIBRATION_STEP1":
            self.calib_orig = (event.x, event.y)
            self.mode = "CALIBRATION_STEP2"
        elif self.mode == "CALIBRATION_STEP2":
            x_align, y_align = event.x, event.y
            self.calib_align = (x_align, y_align) # Guarda o ponto da haste
            self.tracker.set_calibration(self.calib_orig[0], self.calib_orig[1], x_align, y_align)
            self.mode = "PREVIEW"

    def on_closing(self):
        self.tracker.stop_tracking()
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()