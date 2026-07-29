import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import pims
import time
import sys
import matplotlib
import matplotlib.cm as cm

class CinePlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Cine Video Player (PIMS)")
        self.root.geometry("1575x975") 

        # Player State
        self.reader = None
        self.total_frames = 0
        self.current_frame = 0
        self.base_fps = 30 
        self.is_playing = False
        self.speed_multiplier = 1.0
        self.rotation_angle = 0  
        self.is_mirrored = False  
        self._play_job = None
        
        # Interaction & Kill Switch Flags
        self._is_closing = False
        self._interaction_paused = False
        self._interaction_playing_state = False
        
        # Intercept the window native close event ('X' button)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_font()
        self.setup_gui()
        self.force_foreground()

    def force_foreground(self):
        """Forces the window to pop up on top of all other windows on macOS."""
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.focus_force()
        self.root.after(50, lambda: self.root.attributes('-topmost', False))

    def setup_font(self):
        font_size = 28
        try:
            self.font = ImageFont.truetype("Arial.ttf", font_size)
        except OSError:
            try:
                self.font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
            except OSError:
                try:
                    self.font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except OSError:
                    self.font = ImageFont.load_default()

    def setup_gui(self):
        self.controls_frame = tk.Frame(self.root, pady=10, padx=10)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.open_btn = tk.Button(self.controls_frame, text="Open .cine", command=self.open_file)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        
        # New Explicit Close Button

        self.play_btn = tk.Button(self.controls_frame, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.restart_btn = tk.Button(self.controls_frame, text="Restart", command=self.restart, state=tk.DISABLED)
        self.restart_btn.pack(side=tk.LEFT, padx=5)

        self.rotate_btn = tk.Button(self.controls_frame, text="Rotate 90°", command=self.rotate_video, state=tk.DISABLED)
        self.rotate_btn.pack(side=tk.LEFT, padx=5)

        self.mirror_btn = tk.Button(self.controls_frame, text="Mirror", command=self.mirror_video, state=tk.DISABLED)
        self.mirror_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(self.controls_frame, text="Colormap:").pack(side=tk.LEFT, padx=(10, 0))
        self.cmap_var = tk.StringVar(value="Grayscale")
        cmap_options = ["Grayscale", "Viridis", "Plasma", "Inferno", "Magma", "Jet", "Hot"]
        self.cmap_dropdown = tk.OptionMenu(self.controls_frame, self.cmap_var, *cmap_options, command=self.on_cmap_change)
        self.cmap_dropdown.config(state=tk.DISABLED)
        self.cmap_dropdown.pack(side=tk.LEFT, padx=5)

        self.slider = tk.Scale(self.controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                               command=self.on_slider_move, state=tk.DISABLED, showvalue=False)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.frame_label = tk.Label(self.controls_frame, text="0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        tk.Label(self.controls_frame, text="Speed:").pack(side=tk.LEFT, padx=(5, 0))
        
        self.speed_scale = tk.Scale(self.controls_frame, from_=-1.0, to=1.0, resolution=0.01, 
                                    orient=tk.HORIZONTAL, command=self.change_speed, showvalue=False)
        self.speed_scale.set(0.0)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.speed_label = tk.Label(self.controls_frame, text="1.00x", width=5)
        self.speed_label.pack(side=tk.LEFT)

        self.close_btn = tk.Button(self.controls_frame, text="Close App", command=self.on_closing, fg="red")
        self.close_btn.pack(side=tk.LEFT, padx=5)

        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        self.video_frame.bind("<Configure>", self.on_window_resize)
        
        # Apply the pause-on-click interaction to UI elements
        self.make_interactive(self.slider)
        self.make_interactive(self.speed_scale)
        self.make_interactive(self.cmap_dropdown)
        self.make_interactive(self.rotate_btn)
        self.make_interactive(self.mirror_btn)
        self.make_interactive(self.restart_btn)

    def make_interactive(self, widget):
        """Binds mouse press and release events to handle auto-pausing"""
        widget.bind("<ButtonPress-1>", self.pause_for_interaction)
        widget.bind("<ButtonRelease-1>", self.resume_after_interaction)

    def pause_for_interaction(self, event=None):
        if self._is_closing: return
        
        # Lock in the state only once per interaction
        if not self._interaction_paused:
            self._interaction_playing_state = self.is_playing
            self._interaction_paused = True
            
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="Play")
            if self._play_job:
                self.root.after_cancel(self._play_job)
                self._play_job = None

    def resume_after_interaction(self, event=None):
        if self._is_closing: return
        # Schedule the resume shortly after so widget commands have time to fire first
        self.root.after(100, self._do_resume)

    def _do_resume(self):
        if self._is_closing: return
        
        if self._interaction_paused:
            self._interaction_paused = False
            # Only resume if the video was actually playing before they clicked
            if self._interaction_playing_state and not self.is_playing:
                self.is_playing = True
                self.play_btn.config(text="Pause")
                self.play_loop()

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Cine files", "*.cine *.cin"), ("All files", "*.*")])
        if not file_path:
            return

        try:
            if self.reader is not None:
                self.reader.close()

            self.reader = pims.Cine(file_path)
            self.total_frames = len(self.reader)
            self.base_fps = 30

            self.current_frame = 0
            self.rotation_angle = 0 
            self.is_mirrored = False 
            
            self.slider.config(state=tk.NORMAL, to=self.total_frames - 1)
            self.slider.set(0)
            self.speed_scale.set(0.0)
            self.play_btn.config(state=tk.NORMAL, text="Play")
            self.restart_btn.config(state=tk.NORMAL)
            self.rotate_btn.config(state=tk.NORMAL)
            self.mirror_btn.config(state=tk.NORMAL)
            self.cmap_dropdown.config(state=tk.NORMAL)
            self.is_playing = False
            
            self.show_frame(self.current_frame)
            self.force_foreground()

        except Exception as e:
            messagebox.showerror("Error", f"Could not open file via PIMS:\n{str(e)}")

    def toggle_play(self):
        if not self.reader:
            return

        self.is_playing = not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play")
        
        if self.is_playing:
            self.play_loop()
        else:
            if self._play_job:
                self.root.after_cancel(self._play_job)
                self._play_job = None

    def restart(self):
        if not self.reader:
            return
        self.current_frame = 0
        self.slider.set(0)
        self.show_frame(0)

    def rotate_video(self):
        if not self.reader:
            return
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.show_frame(self.current_frame)

    def mirror_video(self):
        if not self.reader:
            return
        self.is_mirrored = not self.is_mirrored
        self.show_frame(self.current_frame)

    def on_cmap_change(self, value):
        self._do_resume()
        if self.reader and not self.is_playing:
            self.show_frame(self.current_frame)

    def on_slider_move(self, val):
        if not self.reader:
            return
        val = int(val)
        if val != self.current_frame:
            self.current_frame = val
            # Update the frame explicitly when manually scrubbing while paused
            if not self.is_playing:
                self.show_frame(self.current_frame)

    def change_speed(self, val):
        log_val = float(val)
        self.speed_multiplier = 10 ** log_val
        self.speed_label.config(text=f"{self.speed_multiplier:.2f}x")

    def on_window_resize(self, event):
        if self.reader and not self.is_playing:
            self.show_frame(self.current_frame)

    def show_frame(self, frame_idx):
        if self._is_closing:
            return

        try:
            frame_data = self.reader[frame_idx]
            
            if frame_data.dtype != np.uint8:
                f_min = frame_data.min()
                f_max = frame_data.max()
                if f_max > f_min:
                    norm_data = (255.0 * (frame_data - f_min) / (f_max - f_min)).astype(np.uint8)
                else:
                    norm_data = frame_data.astype(np.uint8)
            else:
                norm_data = frame_data

            cmap_name = self.cmap_var.get()
            if cmap_name != "Grayscale":
                try:
                    colormap = matplotlib.colormaps[cmap_name.lower()]
                except AttributeError:
                    colormap = cm.get_cmap(cmap_name.lower())
                    
                colored_data = colormap(norm_data / 255.0)
                colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)
                img = Image.fromarray(colored_data, 'RGB')
            else:
                img = Image.fromarray(norm_data)

            if self.is_mirrored:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.rotation_angle != 0:
                img = img.rotate(self.rotation_angle, expand=True)
            
            win_w = self.video_frame.winfo_width()
            win_h = self.video_frame.winfo_height()
            
            if win_w > 10 and win_h > 10:
                img_w, img_h = img.size
                scale = min(win_w / img_w, win_h / img_h) 
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            try:
                first_img_no = self.reader.header_dict['first_image_no']
            except (AttributeError, KeyError):
                first_img_no = 0
            
            capture_fps = getattr(self.reader, 'frame_rate', 1)
            if capture_fps <= 0: capture_fps = 1
            
            current_image_no = first_img_no + frame_idx
            frame_time = (current_image_no / float(capture_fps)) + (1.0 / float(capture_fps))
            
            draw = ImageDraw.Draw(img)
            time_text = f"Time: {frame_time:.6f} s"
            
            x, y = 15, 15
            for offset_x in [-2, -1, 0, 1, 2]:
                for offset_y in [-2, -1, 0, 1, 2]:
                    if offset_x == 0 and offset_y == 0: continue
                    draw.text((x + offset_x, y + offset_y), time_text, fill="black", font=self.font)
            
            draw.text((x, y), time_text, fill="yellow", font=self.font)

            if self._is_closing:
                return

            self.photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.photo)
            
            self.frame_label.config(text=f"{frame_idx} / {self.total_frames - 1}")
            self.root.update_idletasks()
            
        except IndexError:
            self.is_playing = False
            self.play_btn.config(text="Play")

    def play_loop(self):
        if self._is_closing or not self.is_playing:
            return
            
        start_time = time.time()

        self.show_frame(self.current_frame)
        
        if self._is_closing:
            return
            
        self.slider.set(self.current_frame)
        self.current_frame += 1
        
        if self.current_frame >= self.total_frames:
            self.is_playing = False
            self.play_btn.config(text="Play")
            self.current_frame = self.total_frames - 1
            return

        processing_time_ms = (time.time() - start_time) * 1000
        target_delay = (1000 / self.base_fps) / self.speed_multiplier
        actual_delay = max(1, int(target_delay - processing_time_ms))
        
        self._play_job = self.root.after(actual_delay, self.play_loop)

    def on_closing(self):
        """Safely clean up running processes before shutting down the app."""
        self._is_closing = True
        self.is_playing = False
        
        # Cancel the pending loop tasks safely
        if self._play_job:
            try:
                self.root.after_cancel(self._play_job)
                self._play_job = None
            except Exception:
                pass
            
        # Free up the file from memory
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
                
        # Completely terminate Tkinter processes
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = CinePlayer(root)
    root.mainloop()