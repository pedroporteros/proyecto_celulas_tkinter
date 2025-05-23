import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading # Para procesar video sin congelar la GUI
import os
import time # Para controlar el FPS en la reproducción

# --- Configuración del Modelo YOLO ---
MODEL_NAME = 'modelo_celulas_entrenado_yolo_v8.pt' # Asegúrate que este archivo exista
PROCESSED_VIDEO_FILENAME = "processed_video_output.mp4" # Nombre del archivo de video procesado

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detector")
        self.root.geometry("1000x800") # Aumenté un poco más la altura para los controles

        # --- Variables ---
        self.model = None
        self.filepath = None
        self.is_video = False
        self.cap = None # Para VideoCapture durante el procesamiento
        self.video_processing_active = False
        self.detected_classes_set = set()

        self.processed_video_path = None # Ruta al video procesado guardado
        self.video_writer = None # Para guardar el video procesado
        self.original_video_fps = 30 # FPS por defecto, se intentará obtener del video

        self.is_replaying = False # Flag para controlar la reproducción del video procesado
        self.is_paused = False # Flag para pausar la reproducción
        self.replay_cap = None # VideoCapture para la reproducción
        self.total_frames = 0 # Total de frames del video
        self.current_frame_pos = 0 # Posición actual del frame
        self.timeline_updating = False # Flag para evitar bucles al actualizar timeline

        # --- Estilo ---
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", font=('Helvetica', 10))
        style.configure("TLabel", padding=6, font=('Helvetica', 10))
        style.configure("TFrame", padding=10)

        # --- Layout principal ---
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Panel de Control (Izquierda) ---
        control_panel = ttk.Frame(main_frame, width=250)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        control_panel.pack_propagate(False)

        self.btn_load = ttk.Button(control_panel, text="Cargar Imagen/Video", command=self.load_file)
        self.btn_load.pack(pady=10, fill=tk.X)

        self.lbl_filepath = ttk.Label(control_panel, text="Archivo: Ninguno seleccionado", wraplength=230)
        self.lbl_filepath.pack(pady=5, fill=tk.X)

        self.btn_process = ttk.Button(control_panel, text="Procesar", command=self.process_content, state=tk.DISABLED)
        self.btn_process.pack(pady=10, fill=tk.X)

        self.btn_play_processed = ttk.Button(control_panel, text="Reproducir Video Procesado", command=self.start_replay_processed_video, state=tk.DISABLED)
        self.btn_play_processed.pack(pady=10, fill=tk.X)

        self.lbl_status = ttk.Label(control_panel, text="Estado: Listo")
        self.lbl_status.pack(pady=5, fill=tk.X)

        ttk.Separator(control_panel, orient='horizontal').pack(fill='x', pady=10)

        self.lbl_classes_title = ttk.Label(control_panel, text="Clases Detectadas:")
        self.lbl_classes_title.pack(pady=5, anchor='w')

        self.listbox_classes = tk.Listbox(control_panel, height=15, font=('Helvetica', 9))
        self.listbox_classes.pack(fill=tk.BOTH, expand=True)

        # --- Panel de Visualización (Derecha) ---
        display_panel = ttk.Frame(main_frame)
        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lbl_image_display = ttk.Label(display_panel, text="El contenido procesado se mostrará aquí.", anchor="center")
        self.lbl_image_display.pack(fill=tk.BOTH, expand=True)
        self.lbl_image_display.configure(background='lightgrey')

        # --- Controles de Video ---
        self.video_controls_frame = ttk.Frame(display_panel)
        self.video_controls_frame.pack(fill=tk.X, pady=10)
        self.video_controls_frame.pack_forget() # Ocultar inicialmente

        # Botones de control
        controls_buttons_frame = ttk.Frame(self.video_controls_frame)
        controls_buttons_frame.pack(fill=tk.X, pady=5)

        self.btn_play_pause = ttk.Button(controls_buttons_frame, text="▶️", command=self.toggle_play_pause, width=3)
        self.btn_play_pause.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(controls_buttons_frame, text="⏹️", command=self.stop_replay, width=3)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Timeline (barra de progreso)
        timeline_frame = ttk.Frame(self.video_controls_frame)
        timeline_frame.pack(fill=tk.X, pady=5)

        self.timeline_var = tk.DoubleVar()
        self.timeline_scale = ttk.Scale(timeline_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                       variable=self.timeline_var, command=self.on_timeline_change)
        self.timeline_scale.pack(fill=tk.X, padx=5)

        # Etiquetas de tiempo
        time_frame = ttk.Frame(self.video_controls_frame)
        time_frame.pack(fill=tk.X, pady=2)

        self.lbl_current_time = ttk.Label(time_frame, text="00:00", font=('Helvetica', 9))
        self.lbl_current_time.pack(side=tk.LEFT)

        self.lbl_total_time = ttk.Label(time_frame, text="00:00", font=('Helvetica', 9))
        self.lbl_total_time.pack(side=tk.RIGHT)

        # --- Cargar modelo YOLO ---
        self.load_yolo_model()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_yolo_model(self):
        try:
            self.lbl_status.config(text="Estado: Cargando modelo YOLO...")
            self.root.update_idletasks()
            if not os.path.exists(MODEL_NAME):
                messagebox.showerror("Error de Modelo", f"El archivo del modelo '{MODEL_NAME}' no se encuentra. Por favor, verifica la ruta.")
                self.lbl_status.config(text="Error: Modelo no encontrado.")
                self.root.quit()
                return
            self.model = YOLO(MODEL_NAME)
            self.lbl_status.config(text=f"Estado: Modelo {MODEL_NAME} cargado.")
            print(f"Modelo YOLO {MODEL_NAME} cargado exitosamente.")
            self.btn_load.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error de Modelo", f"No se pudo cargar el modelo YOLO: {e}")
            self.lbl_status.config(text="Error: Fallo al cargar modelo.")
            self.root.quit()

    def load_file(self):
        if self.video_processing_active:
            messagebox.showwarning("Procesando", "Hay un video en proceso. Por favor, espere o cierre la ventana.")
            return
        if self.is_replaying:
            self.stop_replay() # Detener reproducción si se está cargando un nuevo archivo

        self.filepath = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=(("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("Archivos de Video", "*.mp4 *.avi *.mov *.mkv"),
                       ("Todos los archivos", "*.*"))
        )
        if self.filepath:
            self.lbl_filepath.config(text=f"Archivo: {os.path.basename(self.filepath)}")
            self.btn_process.config(state=tk.NORMAL)
            self.btn_play_processed.config(state=tk.DISABLED) # Deshabilitar al cargar nuevo archivo
            self.listbox_classes.delete(0, tk.END)
            self.detected_classes_set.clear()
            self.processed_video_path = None # Resetear ruta de video procesado

            ext = os.path.splitext(self.filepath)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                self.is_video = False
                self.display_image_preview(self.filepath)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                self.is_video = True
                self.display_video_preview_frame(self.filepath)
            else:
                messagebox.showerror("Error", "Formato de archivo no soportado.")
                self.filepath = None
                self.btn_process.config(state=tk.DISABLED)
                self.lbl_filepath.config(text="Archivo: Ninguno seleccionado")
        else:
            self.lbl_filepath.config(text="Archivo: Ninguno seleccionado")
            self.btn_process.config(state=tk.DISABLED)
            self.btn_play_processed.config(state=tk.DISABLED)

    def display_image_preview(self, image_data, is_processed_frame=False):
        try:
            if is_processed_frame:
                img = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
            else:
                img = Image.open(image_data)

            panel_width = self.lbl_image_display.winfo_width()
            panel_height = self.lbl_image_display.winfo_height()
            if panel_width < 2 or panel_height < 2:
                panel_width, panel_height = 700, 500 # Ajustar si es necesario

            img.thumbnail((panel_width - 20, panel_height - 20), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.lbl_image_display.config(image=photo, text="")
            self.lbl_image_display.image = photo
        except Exception as e:
            print(f"Error al mostrar imagen/frame: {e}")
            self.lbl_image_display.config(text="Error al mostrar contenido.", image=None)
            self.lbl_image_display.image = None

    def display_video_preview_frame(self, path):
        try:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.display_image_preview(frame, is_processed_frame=True)
            else:
                self.lbl_image_display.config(text="No se pudo leer el primer frame del video.", image=None)
        except Exception as e:
            print(f"Error al mostrar preview de video: {e}")
            self.lbl_image_display.config(text="Error al mostrar preview de video.", image=None)

    def process_content(self):
        if not self.filepath or not self.model:
            messagebox.showwarning("Advertencia", "Por favor, carga un archivo y asegúrate que el modelo YOLO esté cargado.")
            return
        if self.is_replaying:
            self.stop_replay()

        self.btn_process.config(state=tk.DISABLED)
        self.btn_load.config(state=tk.DISABLED)
        self.btn_play_processed.config(state=tk.DISABLED)
        self.lbl_status.config(text="Estado: Procesando...")
        self.root.update_idletasks()
        self.listbox_classes.delete(0, tk.END)
        self.detected_classes_set.clear()
        self.processed_video_path = None # Resetear

        if self.is_video:
            self.video_processing_active = True
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
        else:
            self.process_image()
            self.btn_process.config(state=tk.NORMAL) # Para imagen, se reactiva aquí
            self.btn_load.config(state=tk.NORMAL)

    def process_image(self):
        try:
            results = self.model.predict(source=self.filepath, verbose=False)
            annotated_frame = results[0].plot()
            self.display_image_preview(annotated_frame, is_processed_frame=True)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    self.detected_classes_set.add(class_name)
            self.update_class_list()
            self.lbl_status.config(text="Estado: Imagen procesada.")
        except Exception as e:
            messagebox.showerror("Error de Procesamiento", f"Ocurrió un error al procesar la imagen: {e}")
            self.lbl_status.config(text="Error: Fallo en procesamiento.")
        finally:
            # Reactivar botones es manejado por process_content para imágenes
            pass


    def process_video(self):
        try:
            self.cap = cv2.VideoCapture(self.filepath)
            if not self.cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", "No se pudo abrir el archivo de video."))
                self.root.after(0, self._finalize_video_processing)
                return

            # Obtener propiedades del video para el VideoWriter
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.original_video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.original_video_fps <= 0: self.original_video_fps = 30 # Fallback

            # Configurar VideoWriter
            # Usar un codec común como 'mp4v' para .mp4 o 'XVID' para .avi
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # O 'XVID' si guardas como .avi
            self.processed_video_path = PROCESSED_VIDEO_FILENAME
            self.video_writer = cv2.VideoWriter(self.processed_video_path, fourcc, self.original_video_fps, (frame_width, frame_height))
            
            if not self.video_writer.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", f"No se pudo crear el archivo de video de salida: {self.processed_video_path}"))
                self.root.after(0, self._finalize_video_processing)
                return

            print(f"Guardando video procesado en: {self.processed_video_path} con FPS: {self.original_video_fps}")

            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0

            while self.cap.isOpened() and self.video_processing_active:
                ret, frame = self.cap.read()
                if not ret:
                    break

                results = self.model.predict(source=frame, verbose=False, stream=False)
                annotated_frame = results[0].plot()

                # Escribir frame anotado en el archivo de video de salida
                if self.video_writer:
                    self.video_writer.write(annotated_frame)

                current_frame_classes = set()
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        current_frame_classes.add(class_name)

                self.root.after(0, self.update_video_frame_display, annotated_frame, current_frame_classes)
                processed_frames += 1
                progress_text = f"Estado: Procesando video ({processed_frames}/{total_frames if total_frames > 0 else '?'})..."
                self.root.after(0, self.lbl_status.config, {"text": progress_text})
                # time.sleep(0.001) # Pequeña pausa opcional

            self.cap.release()
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None # Importante para evitar errores al cerrar
                print("Video procesado guardado.")


        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Error de Procesamiento", f"Ocurrió un error durante el procesamiento del video: {e}"))
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release() # Asegurarse de liberar si hay error
                self.video_writer = None
        finally:
            self.root.after(0, self._finalize_video_processing)

    def update_video_frame_display(self, annotated_frame, frame_classes):
        if not self.video_processing_active:
            return
        self.display_image_preview(annotated_frame, is_processed_frame=True)
        new_classes_found = False
        for cls_name in frame_classes:
            if cls_name not in self.detected_classes_set:
                self.detected_classes_set.add(cls_name)
                new_classes_found = True
        if new_classes_found:
            self.update_class_list()

    def _finalize_video_processing(self):
        self.video_processing_active = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        if self.video_writer and self.video_writer.isOpened(): # Por si acaso no se liberó
             self.video_writer.release()
             self.video_writer = None

        self.lbl_status.config(text="Estado: Video procesado (o detenido).")
        self.btn_process.config(state=tk.NORMAL)
        self.btn_load.config(state=tk.NORMAL)
        if self.processed_video_path and os.path.exists(self.processed_video_path):
             self.btn_play_processed.config(state=tk.NORMAL)
        else:
             self.btn_play_processed.config(state=tk.DISABLED)
        self.update_class_list()
        print("Procesamiento de video finalizado.")


    def start_replay_processed_video(self):
        if not self.processed_video_path or not os.path.exists(self.processed_video_path):
            messagebox.showinfo("Información", "No hay video procesado para reproducir o el archivo no existe.")
            return
        if self.video_processing_active:
            messagebox.showwarning("Procesando", "No se puede reproducir mientras se procesa otro video.")
            return
        if self.is_replaying: # Si ya está reproduciendo, detener
            self.stop_replay()
            return

        try:
            self.replay_cap = cv2.VideoCapture(self.processed_video_path)
            if not self.replay_cap.isOpened():
                messagebox.showerror("Error", f"No se pudo abrir el video procesado: {self.processed_video_path}")
                self.replay_cap = None
                return

            # Obtener propiedades del video
            replay_fps = self.replay_cap.get(cv2.CAP_PROP_FPS)
            if replay_fps <= 0 : replay_fps = self.original_video_fps
            if replay_fps <= 0 : replay_fps = 30

            self.total_frames = int(self.replay_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_pos = 0
            self.frame_delay_ms = int(1000 / replay_fps)
            
            # Configurar timeline
            self.timeline_var.set(0)
            self.timeline_scale.config(to=self.total_frames - 1)
            
            # Configurar etiquetas de tiempo
            total_seconds = self.total_frames / replay_fps if replay_fps > 0 else 0
            self.lbl_total_time.config(text=self.format_time(total_seconds))
            self.lbl_current_time.config(text="00:00")
            
            self.is_replaying = True
            self.is_paused = False
            self.btn_load.config(state=tk.DISABLED)
            self.btn_process.config(state=tk.DISABLED)
            self.btn_play_processed.config(text="Detener Video")
            self.btn_play_pause.config(text="⏸️")
            self.lbl_status.config(text="Estado: Reproduciendo video...")
            
            # Mostrar controles de video
            self.video_controls_frame.pack(fill=tk.X, pady=10)
            
            self.replay_frame()

        except Exception as e:
            messagebox.showerror("Error de Reproducción", f"No se pudo iniciar la reproducción: {e}")
            self.stop_replay()

    def replay_frame(self):
        if not self.is_replaying or not self.replay_cap or not self.replay_cap.isOpened():
            self.stop_replay()
            return

        if not self.is_paused:
            ret, frame = self.replay_cap.read()
            if ret:
                self.current_frame_pos = int(self.replay_cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.display_image_preview(frame, is_processed_frame=True)
                self.update_timeline()
                self.root.after(self.frame_delay_ms, self.replay_frame)
            else: # Fin del video
                self.stop_replay()
                self.lbl_status.config(text="Estado: Reproducción finalizada.")
        else:
            # Si está pausado, seguir verificando pero no avanzar frames
            self.root.after(100, self.replay_frame)

    def toggle_play_pause(self):
        if not self.is_replaying:
            return
            
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_play_pause.config(text="▶️")
            self.lbl_status.config(text="Estado: Video pausado.")
        else:
            self.btn_play_pause.config(text="⏸️")
            self.lbl_status.config(text="Estado: Reproduciendo video...")

    def on_timeline_change(self, value):
        if not self.is_replaying or not self.replay_cap or self.timeline_updating:
            return
            
        try:
            frame_pos = int(float(value))
            self.replay_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame_pos = frame_pos
            
            # Leer y mostrar el frame en la nueva posición
            ret, frame = self.replay_cap.read()
            if ret:
                self.display_image_preview(frame, is_processed_frame=True)
                # Retroceder un frame para que la reproducción continúe desde la posición correcta
                self.replay_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            self.update_time_display()
            
        except Exception as e:
            print(f"Error al cambiar posición del timeline: {e}")

    def update_timeline(self):
        if not self.is_replaying or self.timeline_updating:
            return
            
        self.timeline_updating = True
        try:
            self.timeline_var.set(self.current_frame_pos)
            self.update_time_display()
        finally:
            self.timeline_updating = False

    def update_time_display(self):
        if not self.is_replaying or not self.replay_cap:
            return
            
        try:
            fps = self.replay_cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = self.original_video_fps if self.original_video_fps > 0 else 30
                
            current_seconds = self.current_frame_pos / fps
            self.lbl_current_time.config(text=self.format_time(current_seconds))
        except Exception as e:
            print(f"Error al actualizar tiempo: {e}")

    def format_time(self, seconds):
        """Convierte segundos a formato MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def stop_replay(self):
        self.is_replaying = False
        self.is_paused = False
        if self.replay_cap:
            self.replay_cap.release()
            self.replay_cap = None
        
        self.btn_load.config(state=tk.NORMAL)
        self.btn_process.config(state=tk.NORMAL if self.filepath else tk.DISABLED)
        self.btn_play_processed.config(text="Reproducir Video Procesado")
        self.btn_play_pause.config(text="▶️")
        
        # Ocultar controles de video
        self.video_controls_frame.pack_forget()
        
        if not self.video_processing_active:
             self.lbl_status.config(text="Estado: Listo" if not self.filepath else "Estado: Archivo cargado")


    def update_class_list(self):
        self.listbox_classes.delete(0, tk.END)
        sorted_classes = sorted(list(self.detected_classes_set))
        for cls_name in sorted_classes:
            self.listbox_classes.insert(tk.END, cls_name)

    def on_closing(self):
        print("Cerrando aplicación...")
        if self.video_processing_active:
            self.video_processing_active = False
            print("Deteniendo procesamiento de video activo...")
            # El hilo es daemon, se cerrará. Esperar un poco puede ser bueno.
            # Si video_writer está activo, intentar cerrarlo podría ser problemático
            # desde el hilo principal si el hilo de video aún lo usa.
            # La bandera video_processing_active debería ser suficiente para que el hilo termine limpiamente.

        if self.is_replaying:
            self.stop_replay()
            print("Deteniendo reproducción de video activa...")

        # Liberar recursos explícitamente
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release() # Importante
        if self.replay_cap and self.replay_cap.isOpened():
            self.replay_cap.release()

        # Opcional: eliminar el video procesado temporal al cerrar
        # if self.processed_video_path and os.path.exists(self.processed_video_path):
        #     try:
        #         os.remove(self.processed_video_path)
        #         print(f"Video procesado temporal '{self.processed_video_path}' eliminado.")
        #     except Exception as e:
        #         print(f"No se pudo eliminar '{self.processed_video_path}': {e}")

        self.root.after(100, self.root.destroy) # Dar un pequeño margen para que los hilos finalicen


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()