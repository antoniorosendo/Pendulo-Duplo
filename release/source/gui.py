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

        # Configuração da Janela
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.title("SISTEMA DE RASTREAMENTO - PENDULO DUPLO")
        self.geometry("900x600")

        # Inicializa o motor C++
        self.tracker = pendulum_tracker.PendulumTracker()
        
        # Inicia a câmera IMEDIATAMENTE ao abrir o programa
        self.cap = cv2.VideoCapture(0)
        
        # Variáveis de Estado
        self.mode = "PREVIEW" # Estados: "PREVIEW", "COUNTDOWN", "TRACKING"
        self.countdown_value = 3

        # Histórico do pêndulo
        self.history_x1, self.history_y1 = [], []
        self.history_x2, self.history_y2 = [], []

        # Histórico de Hardware
        self.start_time = 0
        self.time_history = []

        # Armazenamento das threads
        MAX_THREADS = 8   # arquitetura do projeto: 1 captura + 6 processamento + 1 escrita
        self.process = psutil.Process(os.getpid())

        # O TID identifica uma thread do sistema operacional.
        self.main_tid = threading.get_native_id()

        self.thread_usage = {}
        self.previous_thread_times = {}
        self.previous_sample_time = time.perf_counter()
        
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

        self.btn_plot = ctk.CTkButton(self.sidebar, text="Gráfico do Pêndulo", fg_color="blue", command=self.plot_graph)
        self.btn_plot.pack(padx=20, pady=10)

        # Novo botão para o gráfico das threads
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

        # Área de Exibição do Vídeo
        self.video_container = ctk.CTkFrame(self)
        self.video_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(self.video_container, text="")
        self.video_label.pack(fill="both", expand=True)

        # Tranca de segurança para liberar a câmera ao fechar a janela (evita processos fantasmas)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Inicia os loops
        self.update_system_stats()
        self.update_frame()

    def start_video(self):
        if self.mode == "PREVIEW":
            # Limpa os dados físicos
            self.history_x1.clear()
            self.history_y1.clear()
            self.history_x2.clear()
            self.history_y2.clear()
            
            # Limpa os dados de hardware
            self.time_history.clear()
            for th in self.thread_history:
                th.clear()

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

    def stop_video(self):
        if self.mode == "TRACKING" and len(self.time_history) > 0:
            self.save_threads_csv()
            self.save_pendulum_csv()

        self.mode = "PREVIEW"
        
    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                
                # --- MÁQUINA DE ESTADOS VISUAL ---
                if self.mode == "PREVIEW":
                    cv2.putText(frame, "PREVIEW - ALINHE O PENDULO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                elif self.mode == "COUNTDOWN":
                    # Desenha o número gigante no centro da tela
                    cv2.putText(frame, str(self.countdown_value), (280, 280), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 165, 255), 10)
                    cv2.putText(frame, "PREPARANDO...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                elif self.mode == "TRACKING":
                    cv2.putText(frame, "GRAVANDO!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 1. Matemática do Motor C++
                    x1, y1, x2, y2 = self.tracker.track_frame(frame)

                    # 2. Desenho das miras e salvamento
                    box = 25 
                    if x1 > 0 and y1 > 0:
                        self.history_x1.append(x1)
                        self.history_y1.append(-y1)
                        cv2.rectangle(frame, (int(x1)-box, int(y1)-box), (int(x1)+box, int(y1)+box), (0, 0, 255), 2)
                        cv2.circle(frame, (int(x1), int(y1)), 4, (0, 255, 0), -1) 
                        
                    if x2 > 0 and y2 > 0:
                        self.history_x2.append(x2)
                        self.history_y2.append(-y2)
                        cv2.rectangle(frame, (int(x2)-box, int(y2)-box), (int(x2)+box, int(y2)+box), (255, 0, 0), 2)
                        cv2.circle(frame, (int(x2), int(y2)), 4, (0, 255, 0), -1)

                # Atualiza a interface gráfica com o frame pronto
                cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_img)
                img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk

        # Roda o loop o tempo todo, lendo a câmera ininterruptamente
        self.after(15, self.update_frame)

    def update_system_stats(self):
        self.collect_thread_usage()
    
        for i, tid in enumerate(self.thread_ids):
            usage = self.thread_usage.get(tid, 0.0)
    
            self.thread_labels[i].configure(
                text=f"T{i}: {usage:.1f}%"
            )
    
            color = "#ff3333" if usage > 70 else "#00a86b"
            self.thread_bars[i].configure(progress_color=color)
            self.thread_bars[i].set(usage / 100.0)
    
            if self.mode == "TRACKING":
                self.thread_history[i].append(usage)
    
        self.after(500, self.update_system_stats)

    def collect_thread_usage(self):
        now = time.perf_counter()
        elapsed = now - self.previous_sample_time

        if elapsed <= 0:
            return

        current = {
            thread.id: thread.user_time + thread.system_time
            for thread in self.process.threads()
        }

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
        if not self.history_x1 and not self.history_x2:
            return 
        plt.style.use('dark_background')
        plt.figure(figsize=(8, 6))
        if self.history_x1:
            plt.plot(self.history_x1, self.history_y1, color='red', label='Massa 1')
        if self.history_x2:
            plt.plot(self.history_x2, self.history_y2, color='blue', label='Massa 2')
        plt.title('TRAJETORIA DO PENDULO DUPLO')
        plt.xlabel('Posição X')
        plt.ylabel('Posição Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_threads(self):
        if not self.time_history:
            return 
            
        plt.style.use('dark_background')
        plt.figure(figsize=(8, 6))
        
        cmap = plt.get_cmap("tab10")
        for i in range(self.num_threads):
            if self.thread_history[i]:
                plt.plot(self.time_history, self.thread_history[i], color=cmap(i), label=f'Thread {i}')
                
        plt.title('GRAFICO DE USO DAS THREADS NO TEMPO')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Uso da CPU (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def save_pendulum_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../../CSV/pendulo/pendulo_log_{timestamp}.csv"
        
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=';') 
                
                # Cabeçalho idêntico ao padrão utilizado no C++
                writer.writerow(["time", "x1", "y1", "x2", "y2"])
                
                # Garante que vai iterar pelo menor tamanho de lista para evitar erros
                limit = min(len(self.time_history), len(self.history_x1), len(self.history_x2))
                
                for i in range(limit):
                    t = f"{self.time_history[i]:.2f}".replace('.', ',')
                    # Retira o sinal negativo que foi usado apenas para inverter o gráfico visualmente
                    x1 = self.history_x1[i]
                    y1 = abs(self.history_y1[i]) 
                    x2 = self.history_x2[i]
                    y2 = abs(self.history_y2[i])
                    
                    writer.writerow([t, x1, y1, x2, y2])
                    
            print(f"Sucesso: Arquivo do pendulo salvo em {filename}")
        except Exception as e:
            print(f"Erro ao salvar CSV do pendulo: {e}")

    def save_threads_csv(self):
        # Cria um nome de arquivo único baseado na data e hora atual
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../../CSV/threads/threads_log_{timestamp}.csv"
        
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=';') # Ponto e vírgula facilita abrir no Excel em PT-BR
                
                # Escreve o cabeçalho
                writer.writerow(["Tempo (s)"] + [f"Thread {i} (%)" for i in range(self.num_threads)])

                for i in range(len(self.time_history)):
                    t = f"{self.time_history[i]:.2f}".replace('.', ',')
                    row = [t]
                    for k in range(self.num_threads):
                        row.append(self.thread_history[k][i] if i < len(self.thread_history[k]) else 0)
                    writer.writerow(row)
                    
            print(f"Sucesso: Arquivo de threads salvo em {filename}")
        except Exception as e:
            print(f"Erro ao salvar CSV: {e}")

    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()