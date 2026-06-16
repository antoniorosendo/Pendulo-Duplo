import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import numpy as np
import pendulum_tracker # Importa o seu binário C++ compilado!

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuração da Janela em Modo Escuro
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.title("Sistema de Rastreamento - Pêndulo Duplo")
        self.geometry("900x600")

        # Inicializa o motor C++ passando os caminhos dos templates
        self.tracker = pendulum_tracker.PendulumTracker("../../images/template1.jpg", "../../images/template2.jpg")
        
        self.cap = None
        self.is_tracking = False

        # --- LAYOUT ---
        # Painel Lateral de Controle
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.title_label = ctk.CTkLabel(self.sidebar, text="Controles", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        self.btn_play = ctk.CTkButton(self.sidebar, text="PLAY", fg_color="green", command=self.start_video)
        self.btn_play.pack(padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="STOP", fg_color="red", command=self.stop_video)
        self.btn_stop.pack(padx=20, pady=10)

        # Área de Exibição do Vídeo
        self.video_container = ctk.CTkFrame(self)
        self.video_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(self.video_container, text="Câmera desligada")
        self.video_label.pack(fill="both", expand=True)

    def start_video(self):
        if not self.is_tracking:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.is_tracking = True
            self.update_frame()

    def stop_video(self):
        self.is_tracking = False
        if self.cap:
            self.cap.release()
        self.video_label.configure(image=None, text="Execução encerrada com sucesso.")
        
    def update_frame(self):
        if self.is_tracking and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 1. Envia o frame para o motor C++ processar a matemática pesada
                x1, y1, x2, y2 = self.tracker.track_frame(frame)

                # 2. Python desenha as "miras" (Retângulos e Centro)
                box = 25 # Tamanho do retângulo a partir do centro
                
                if x1 > 0 and y1 > 0:
                    # Massa 1 (Vermelho)
                    cv2.rectangle(frame, (int(x1)-box, int(y1)-box), (int(x1)+box, int(y1)+box), (0, 0, 255), 2)
                    cv2.circle(frame, (int(x1), int(y1)), 4, (0, 255, 0), -1) 
                    
                if x2 > 0 and y2 > 0:
                    # Massa 2 (Azul)
                    cv2.rectangle(frame, (int(x2)-box, int(y2)-box), (int(x2)+box, int(y2)+box), (255, 0, 0), 2)
                    cv2.circle(frame, (int(x2), int(y2)), 4, (0, 255, 0), -1)

                # Converte o frame do OpenCV para formato legível no Tkinter
                cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_img)
                img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk

            # Agenda a atualização do próximo quadro (equivalente ao waitKey do OpenCV)
            self.after(15, self.update_frame)

if __name__ == "__main__":
    app = App()
    app.mainloop()