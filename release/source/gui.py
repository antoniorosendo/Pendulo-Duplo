import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import numpy as np
import pendulum_tracker 
import matplotlib.pyplot as plt
import psutil

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuração da Janela em Modo Escuro
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.title("SISTEMA DE RASTREAMENTO - PENDULO DUPLO")
        self.geometry("900x600")

        # Inicializa o motor C++ passando os caminhos dos templates
        self.tracker = pendulum_tracker.PendulumTracker("../../images/template1.jpg", "../../images/template2.jpg")
        
        self.cap = None
        self.is_tracking = False

        # Listas para guardar o histórico do movimento
        self.history_x1, self.history_y1 = [], []
        self.history_x2, self.history_y2 = [], []

        # --- LAYOUT ---
        # Painel Lateral de Controle
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.title_label = ctk.CTkLabel(self.sidebar, text="CONTROLES", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        self.btn_play = ctk.CTkButton(self.sidebar, text="PLAY", fg_color="green", command=self.start_video)
        self.btn_play.pack(padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="STOP", fg_color="red", command=self.stop_video)
        self.btn_stop.pack(padx=20, pady=10)

        # Botão para gerar o gráfico
        self.btn_plot = ctk.CTkButton(self.sidebar, text="Gerar Gráfico", fg_color="blue", command=self.plot_graph)
        self.btn_plot.pack(padx=20, pady=10)

        # --- PAINEL DE MONITORAMENTO DE THREADS ---
        # Título do monitoramento
        self.lbl_threads = ctk.CTkLabel(self.sidebar, text="USO DE THREADS", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_threads.pack(padx=20, pady=(30, 5))

        self.thread_bars = []
        self.thread_labels = []
        
        # Cria 4 barras de progresso dinâmicas
        for i in range(4):
            frame_bar = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            frame_bar.pack(fill="x", padx=10, pady=2)
            
            lbl = ctk.CTkLabel(frame_bar, text=f"T{i}: 0%", width=45, anchor="w")
            lbl.pack(side="left")
            
            bar = ctk.CTkProgressBar(frame_bar, width=100, progress_color="#00a86b")
            bar.set(0)
            bar.pack(side="right", padx=5)
            
            self.thread_labels.append(lbl)
            self.thread_bars.append(bar)

        # Inicia o loop de monitoramento de hardware
        self.update_system_stats()

        # Área de Exibição do Vídeo
        self.video_container = ctk.CTkFrame(self)
        self.video_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(self.video_container, text="Câmera desligada")
        self.video_label.pack(fill="both", expand=True)

    def start_video(self):
        if not self.is_tracking:
            self.history_x1.clear()
            self.history_y1.clear()
            self.history_x2.clear()
            self.history_y2.clear()

            # self.cap = cv2.VideoCapture(0) 
            self.cap = cv2.VideoCapture("../../videos/teste.mp4")
            self.is_tracking = True
            self.update_frame()

    def stop_video(self):
        self.is_tracking = False
        if self.cap:
            self.cap.release()
        self.video_label.configure(image=None, text="Execução encerrada com sucesso.")
        
    def update_frame(self):
        # O loop agora depende apenas da variavel is_tracking
        if self.is_tracking: 
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # 1. Envia o frame para o motor C++ processar a matematica
                    x1, y1, x2, y2 = self.tracker.track_frame(frame)

                    # 2. Python desenha as miras e salva o historico
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

                    # Atualiza a imagem na interface
                    cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2_img)
                    img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                    
                    self.video_label.configure(image=img_tk, text="")
                    self.video_label.image = img_tk

            # Agenda a atualizacao INDEPENDENTE se o frame atual veio vazio ou nao
            self.after(15, self.update_frame)

    def update_system_stats(self):
        # Lê o uso atual de cada núcleo lógico do processador
        cpu_percentages = psutil.cpu_percent(percpu=True)
        
        # Atualiza as barras de acordo com a porcentagem (limitado a 4 barras para a interface)
        for i in range(min(4, len(cpu_percentages))):
            usage = cpu_percentages[i]
            
            # Atualiza o texto e muda a cor se passar de 70% (fica vermelho)
            self.thread_labels[i].configure(text=f"T{i}: {int(usage)}%")
            color = "#ff3333" if usage > 70 else "#00a86b"
            self.thread_bars[i].configure(progress_color=color)
            
            # A barra do CustomTkinter vai de 0.0 a 1.0
            self.thread_bars[i].set(usage / 100.0)
            
        # Agenda a próxima leitura do processador
        self.after(500, self.update_system_stats)

    def plot_graph(self):
        # Só gera o gráfico se houver dados capturados
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

if __name__ == "__main__":
    app = App()
    app.mainloop()