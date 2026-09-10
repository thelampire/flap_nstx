#!/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import os
import pims
import time
import matplotlib
import matplotlib.cm as cm
try:
    import cv2
    cv2_installed=True
except:
    cv2 = None
    print('OpenCV not installed. True color won\'t be available')
    cv2_installed=False

# Local storage of the NSTX cine files on the server. The files are organized as
# <DATAPATH>/<camera folder>/<year>/<file prefix><shot>.cin
DATAPATH = os.getenv('NSTX_CINE_DATAPATH', '/p/nstxcam')
# Shots before 2010 are stored in the archive directory.
ARCHIVE_DATAPATH = DATAPATH + '-archive'

# Human readable camera aliases -> (camera folder, file name prefix)
CAMERA_ALIASES = {
    'GPI (Phantom710-9205)': ('Phantom710-9205', 'nstx_5_'),
    'Phantom71-5040': ('Phantom71-5040', 'nstx_0_'),
    'Phantom710-9206': ('Phantom710-9206', 'nstx_1_'),
    'Phantom73-6747': ('Phantom73-6747', 'nstx_2_'),
    'Phantom73-6663': ('Phantom73-6663', 'nstx_3_'),
    'Phantom73-8032': ('Phantom73-8032', 'nstx_4_'),
    'Miro4-9373': ('Miro4-9373', 'nstx_6_'),
    'PlasmaTV (Miro2-7988)': ('Miro2-7988', 'nstx_2_'),
    }
DEFAULT_CAMERA = 'GPI (Phantom710-9205)'


def shot_to_year(shot):
    """Return the NSTX campaign year belonging to a shot number."""
    if shot < 118929:
        return 2005
    if shot < 122270:
        return 2006
    if shot < 126511:
        return 2007
    if shot < 131565:
        return 2008
    if shot < 137110:
        return 2009
    return 2010


def cine_file_name(camera_alias, shot):
    """File name of the cine file of a given camera and shot."""
    prefix = CAMERA_ALIASES[camera_alias][1]
    if shot_to_year(shot) < 2006:
        return 'nstx' + str(shot) + '.cin'
    return prefix + str(shot) + '.cin'


def cine_path(camera_alias, shot):
    """Full path of the cine file of a given camera and shot."""
    year = shot_to_year(shot)
    datapath = DATAPATH if year > 2009 else ARCHIVE_DATAPATH
    folder = CAMERA_ALIASES[camera_alias][0]
    return os.path.join(datapath, folder, str(year),
                        cine_file_name(camera_alias, shot))


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
        # Incremented every time playback is stopped so that stale play_loop
        # chains scheduled with root.after() die instead of running in parallel.
        self._play_generation = 0
        # Size of the video area, kept up to date by the <Configure> handler so
        # that show_frame() doesn't have to query Tk for it on every frame.
        self._display_size = None
        # Cache of the 256 entry uint8 colormap lookup tables.
        self._lut_cache = {}
        # Reused Tk image buffer, see _blit().
        self.photo = None
        self._photo_mode = None
        # Re-entrancy guard for show_frame().
        self._rendering = False
        # Set NSTX_CINE_PLAYER_PROFILE=1 to print the per frame render time.
        self._profile = os.getenv('NSTX_CINE_PLAYER_PROFILE', '') not in ('', '0')
        
        # Interaction & Kill Switch Flags
        self._is_closing = False
        self._interaction_paused = False
        self._interaction_playing_state = False
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_font()
        self.setup_gui()
        self.warm_up_colormaps()
        self.force_foreground()

    def warm_up_colormaps(self):
        """Build every colormap lookup table up front.

        The first access to a matplotlib colormap initialises the whole
        colormap registry, which takes a noticeable moment. Doing it while the
        window is starting up keeps the first colormap change as fast as the
        later ones.
        """
        for name in getattr(self, 'cmap_options', ()):
            if name in ('Grayscale', 'True Color (Debayer)'):
                continue
            try:
                self._get_lut(name)
            except Exception:
                pass

    def force_foreground(self):
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
        # Left hand side column holding every button and selector
        self.sidebar_frame = tk.Frame(self.root, pady=10, padx=10)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Right hand side: the video with the playback controls underneath it
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.controls_frame = tk.Frame(self.main_frame, pady=10, padx=10)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.video_frame = tk.Frame(self.main_frame, bg="black")
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.video_frame.pack_propagate(False)

        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH)

        self.video_frame.bind("<Configure>", self.on_window_resize)

        # --- Sidebar widgets ---
        self.open_btn = tk.Button(self.sidebar_frame, text="Open .cine", command=self.open_file)
        self.open_btn.pack(side=tk.TOP, fill=tk.X, pady=3)

        # Name of the file opened through the "Open .cine" button. The space is
        # reserved even when it is empty so the sidebar doesn't jump around.
        self.file_name_var = tk.StringVar(value="")
        self.file_name_label = tk.Label(self.sidebar_frame,
                                        textvariable=self.file_name_var,
                                        anchor="w", justify=tk.LEFT,
                                        wraplength=200, height=2, fg="gray25")
        self.file_name_label.pack(side=tk.TOP, fill=tk.X)

        tk.Label(self.sidebar_frame, text="Camera:", anchor="w").pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        self.camera_var = tk.StringVar(value=DEFAULT_CAMERA)
        self.camera_dropdown = ttk.Combobox(self.sidebar_frame,
                                            textvariable=self.camera_var,
                                            values=list(CAMERA_ALIASES.keys()),
                                            state="readonly", width=22)
        self.camera_dropdown.pack(side=tk.TOP, fill=tk.X, pady=3)
        self.camera_dropdown.bind("<<ComboboxSelected>>", self.on_camera_change)

        tk.Label(self.sidebar_frame, text="Shot:", anchor="w").pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        self.shot_var = tk.StringVar()
        self.shot_entry = tk.Entry(self.sidebar_frame, textvariable=self.shot_var)
        self.shot_entry.pack(side=tk.TOP, fill=tk.X, pady=3)
        self.shot_entry.bind("<Return>", lambda event: self.open_shot())

        self.load_shot_btn = tk.Button(self.sidebar_frame, text="Load shot", command=self.open_shot)
        self.load_shot_btn.pack(side=tk.TOP, fill=tk.X, pady=3)

        self.rotate_btn = tk.Button(self.sidebar_frame, text="Rotate 90°", command=self.rotate_video, state=tk.DISABLED)
        self.rotate_btn.pack(side=tk.TOP, fill=tk.X, pady=(10, 3))

        self.mirror_btn = tk.Button(self.sidebar_frame, text="Mirror", command=self.mirror_video, state=tk.DISABLED)
        self.mirror_btn.pack(side=tk.TOP, fill=tk.X, pady=3)

        tk.Label(self.sidebar_frame, text="Color/Map:", anchor="w").pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        self.cmap_var = tk.StringVar(value="Grayscale")
        if cv2_installed:
            cmap_options = ["Grayscale", "True Color (Debayer)", "Viridis", "Plasma", "Inferno", "Magma", "Jet", "Hot"]
        else:
            cmap_options = ["Grayscale", "Viridis", "Plasma", "Inferno", "Magma", "Jet", "Hot"]

        self.cmap_options = cmap_options
        self.cmap_dropdown = tk.OptionMenu(self.sidebar_frame, self.cmap_var, *cmap_options, command=self.on_cmap_change)
        self.cmap_dropdown.config(state=tk.DISABLED)
        self.cmap_dropdown.pack(side=tk.TOP, fill=tk.X, pady=3)

        self.close_btn = tk.Button(self.sidebar_frame, text="Close App", command=self.on_closing, fg="red")
        self.close_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=3)

        # --- Playback controls under the video ---
        self.play_btn = tk.Button(self.controls_frame, text="Play", width=6, command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.restart_btn = tk.Button(self.controls_frame, text="Restart", command=self.restart, state=tk.DISABLED)
        self.restart_btn.pack(side=tk.LEFT, padx=5)

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

        self.make_interactive(self.slider)
        self.make_interactive(self.speed_scale)
        self.make_interactive(self.cmap_dropdown)
        self.make_interactive(self.rotate_btn)
        self.make_interactive(self.mirror_btn)
        self.make_interactive(self.restart_btn)
        self.make_interactive(self.load_shot_btn)

    def make_interactive(self, widget):
        widget.bind("<ButtonPress-1>", self.pause_for_interaction)
        widget.bind("<ButtonRelease-1>", self.resume_after_interaction)

    def _stop_playback(self):
        """Stop playback and invalidate any play_loop chain still scheduled."""
        self.is_playing = False
        self._play_generation += 1
        if self._play_job is not None:
            try:
                self.root.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None

    def _start_playback(self):
        """Start a single play_loop chain, killing any previous one."""
        self._stop_playback()
        self.is_playing = True
        self.play_btn.config(text="Pause")
        self.play_loop(self._play_generation)

    def pause_for_interaction(self, event=None):
        if self._is_closing: return
        
        if not self._interaction_paused:
            self._interaction_playing_state = self.is_playing
            self._interaction_paused = True
            
        if self.is_playing:
            self._stop_playback()
            self.play_btn.config(text="Play")

    def resume_after_interaction(self, event=None):
        if self._is_closing: return
        self.root.after(100, self._do_resume)

    def _do_resume(self):
        if self._is_closing: return
        
        if self._interaction_paused:
            self._interaction_paused = False
            if self._interaction_playing_state and not self.is_playing:
                self._start_playback()

    def on_camera_change(self, event=None):
        if 'PlasmaTV' in self.camera_var.get() and cv2_installed:
            self.cmap_var.set('True Color (Debayer)')
            if self.reader and not self.is_playing:
                self.show_frame(self.current_frame)

    def open_file(self):
        file_path = filedialog.askopenfilename(initialdir="/p/nstxcam",
                                                filetypes=[("Cine files", "*.cine *.cin"),
                                                           ("All files", "*.*")])
        if not file_path:
            return
        if self.load_file(file_path):
            self.file_name_var.set(os.path.basename(file_path))

    def open_shot(self):
        """Open the cine file of the selected camera and shot number."""
        shot_text = self.shot_var.get().strip()
        if not shot_text:
            messagebox.showerror("Error", "Please specify a shot number.")
            return
        try:
            shot = int(shot_text)
        except ValueError:
            messagebox.showerror("Error", "The shot number should be an integer.")
            return

        file_path = cine_path(self.camera_var.get(), shot)
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found:\n{file_path}")
            return
        if self.load_file(file_path):
            # The shot was loaded from the database, so the name of the
            # previously opened file doesn't belong to the screen any more.
            self.file_name_var.set("")

    def _set_controls_state(self, state):
        """Enable or disable every widget that needs a loaded video."""
        for widget in (self.slider, self.play_btn, self.restart_btn,
                       self.rotate_btn, self.mirror_btn, self.cmap_dropdown):
            widget.config(state=state)

    def load_file(self, file_path):
        try:
            # Any playback chain of the previously loaded file must be killed
            # before swapping the reader, otherwise the old play_loop keeps
            # running next to the new one and the player becomes sluggish.
            self._stop_playback()

            # Reading the file takes a while, so the controls of the previous
            # video are switched off first. They are switched back on once the
            # first frame is actually on the screen.
            self._set_controls_state(tk.DISABLED)
            self.root.update_idletasks()

            if self.reader is not None:
                self.reader.close()

            self.reader = pims.Cine(file_path)

            if 3 not in self.reader.shape and self.reader.setup_fields_dict.get('cfa', 0) != 0:
                self.reader.setup_fields_dict['cfa'] = 0
                self.reader.header_dict['compression'] = 0

            self.total_frames = len(self.reader)
            self.base_fps = 30

            self.current_frame = 0
            self.rotation_angle = 0 
            self.is_mirrored = False 
            
            self.slider.config(to=self.total_frames - 1)
            self.slider.set(0)
            self.speed_scale.set(0.0)
            self.play_btn.config(text="Play")
            self.is_playing = False
            # The "Load shot" button is an interactive widget, so its
            # ButtonRelease would otherwise resume playback of the freshly
            # loaded file behind our back.
            self._interaction_paused = False
            self._interaction_playing_state = False
            
            # The frame size changes with the file, so the reused Tk image
            # buffer has to be dropped and the geometry re-read once.
            self.photo = None
            self._photo_mode = None
            self.video_label.config(image='')
            self.root.update_idletasks()
            self._display_size = None

            self.show_frame(self.current_frame)

            # The video is decoded and visible now, so the controls belonging
            # to it can be used.
            self._set_controls_state(tk.NORMAL)
            self.force_foreground()
            return True

        except Exception as e:
            # Nothing is displayed, so the controls stay switched off.
            self._set_controls_state(tk.DISABLED)
            messagebox.showerror("Error", f"Could not open file via PIMS:\n{str(e)}")
            return False

    def toggle_play(self):
        if not self.reader:
            return

        if not self.is_playing:
            self._start_playback()
        else:
            self._stop_playback()
            self.play_btn.config(text="Play")

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
            if not self.is_playing:
                self.show_frame(self.current_frame)

    def change_speed(self, val):
        log_val = float(val)
        self.speed_multiplier = 10 ** log_val
        self.speed_label.config(text=f"{self.speed_multiplier:.2f}x")

    def on_window_resize(self, event):
        # Cache the new size instead of asking Tk for it on every single frame.
        # winfo_width()/winfo_height() flush the geometry manager, which is one
        # of the expensive calls in the playback loop.
        self._display_size = (event.width, event.height)
        if self.reader and not self.is_playing:
            self.show_frame(self.current_frame)

    def _get_display_size(self):
        """Size of the video area, asking Tk only when it isn't known yet.

        Tk reports 1x1 for a widget whose geometry hasn't been calculated yet.
        That happens for every frame shown before the window is fully mapped,
        and the old code silently skipped the downscaling in that case, so the
        full sensor resolution was pushed into the label on every tick. That is
        the slow state; it used to disappear only when some other action (like
        opening the file dialog) finally pumped the event loop and triggered a
        <Configure>. Force the geometry to be calculated instead of guessing.
        """
        if self._display_size is None or self._display_size[0] <= 10 or self._display_size[1] <= 10:
            self.video_frame.update_idletasks()
            size = (self.video_frame.winfo_width(), self.video_frame.winfo_height())
            if size[0] <= 10 or size[1] <= 10:
                # Still not mapped: fall back to the requested size of the
                # widget rather than rendering at full sensor resolution.
                size = (max(self.video_frame.winfo_reqwidth(), 640),
                        max(self.video_frame.winfo_reqheight(), 480))
                return size
            self._display_size = size
        return self._display_size

    def _get_lut(self, cmap_name):
        """256x3 uint8 lookup table of a matplotlib colormap (cached)."""
        lut = self._lut_cache.get(cmap_name)
        if lut is None:
            try:
                colormap = matplotlib.colormaps[cmap_name.lower()]
            except AttributeError:
                colormap = cm.get_cmap(cmap_name.lower())
            ramp = np.linspace(0., 1., 256)
            lut = (colormap(ramp)[:, :3] * 255).astype(np.uint8)
            self._lut_cache[cmap_name] = lut
        return lut

    def _blit(self, img):
        """Push a PIL image into the label, reusing the Tk image buffer.

        Creating a new PhotoImage for every frame makes Tk allocate and free a
        full size image on each tick and forces the label to renegotiate its
        geometry. Reusing one buffer of a constant size avoids both.

        The buffer has a fixed mode, so it can only be reused while the mode of
        the image stays the same. Pasting an RGB image into a buffer created
        from a grayscale one silently converts it back to grayscale, which is
        why the mode is part of the check.
        """
        if (self.photo is not None
                and self._photo_mode == img.mode
                and (self.photo.width(), self.photo.height()) == img.size):
            self.photo.paste(img)
        else:
            self.photo = ImageTk.PhotoImage(image=img)
            self._photo_mode = img.mode
            self.video_label.config(image=self.photo)

    def show_frame(self, frame_idx):
        if self._is_closing:
            return

        # Resizing the label can dispatch a <Configure> event that calls
        # show_frame() again while it is still running. Without this guard the
        # renders nest and every frame costs a multiple of what it should.
        if self._rendering:
            return
        self._rendering = True
        start_time = time.time()
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
            
            if cmap_name == "True Color (Debayer)":
                try:
                    # SAFETY CHECK: Only debayer if the image is strictly 2D. 
                    # np.squeeze() removes any ghost dimensions (like 1280x720x1).
                    flat_data = np.squeeze(norm_data)

                    if flat_data.ndim == 2:
                        colored_data = cv2.cvtColor(flat_data, cv2.COLOR_BAYER_GB2BGR)
                        img = Image.fromarray(colored_data, 'RGB')
                    else:
                        # If it is already a 3D matrix (e.g., standard color video), just pass it
                        if norm_data.shape[-1] == 3:
                            img = Image.fromarray(norm_data, 'RGB')
                        else:
                            img = Image.fromarray(norm_data)
                except Exception as e:
                    print(f"Debayer fallback due to error: {e}")
                    img = Image.fromarray(norm_data)
                    
            elif cmap_name != "Grayscale":
                # Ensure the data is strictly 2D before applying Matplotlib colormaps
                flat_data = np.squeeze(norm_data)
                if flat_data.ndim == 2:
                    # A uint8 lookup table is far cheaper than calling the
                    # colormap on a float image of the full frame size.
                    colored_data = self._get_lut(cmap_name)[flat_data]
                    img = Image.fromarray(colored_data, 'RGB')
                else:
                    # It already has colors; just display it
                    img = Image.fromarray(norm_data)
            else:
                img = Image.fromarray(norm_data)

            # Apply orientation modifications
            if self.is_mirrored:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.rotation_angle != 0:
                img = img.rotate(self.rotation_angle, expand=True)
            
            # Scale dynamically to window
            win_w, win_h = self._get_display_size()
            
            if win_w > 10 and win_h > 10:
                img_w, img_h = img.size
                scale = min(win_w / img_w, win_h / img_h) 
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            # Timestamp Logic
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
            # One stroked draw instead of 25 separate text renderings.
            try:
                draw.text((x, y), time_text, fill="yellow", font=self.font,
                          stroke_width=2, stroke_fill="black")
            except TypeError:
                # Pillow older than 6.2 doesn't support stroke_width.
                for offset_x in (-2, 0, 2):
                    for offset_y in (-2, 0, 2):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        draw.text((x + offset_x, y + offset_y), time_text,
                                  fill="black", font=self.font)
                draw.text((x, y), time_text, fill="yellow", font=self.font)

            if self._is_closing:
                return

            self._blit(img)

            self.frame_label.config(text=f"{frame_idx} / {self.total_frames - 1}")

            # During playback the loop runs from root.after(), so Tk repaints
            # between the ticks on its own and flushing here would only cost
            # time. A single frame drawn after a colormap change, a rotation or
            # a slider move is not followed by any event though, so without an
            # explicit flush it only appears when Tk next happens to go idle.
            if not self.is_playing:
                self.video_label.update_idletasks()

        except IndexError:
            self._stop_playback()
            self.play_btn.config(text="Play")
        finally:
            self._rendering = False
            if self._profile:
                print(f"frame {frame_idx}: {(time.time() - start_time)*1000:.1f} ms")

    def play_loop(self, generation=None):
        if generation is None:
            generation = self._play_generation
        # A stale chain (from a previously loaded file or an earlier play
        # session) carries an outdated generation and stops here.
        if self._is_closing or not self.is_playing or generation != self._play_generation:
            return
            
        start_time = time.time()

        self.show_frame(self.current_frame)
        
        if self._is_closing or generation != self._play_generation:
            return
            
        self.slider.set(self.current_frame)
        self.current_frame += 1
        
        if self.current_frame >= self.total_frames:
            self._stop_playback()
            self.play_btn.config(text="Play")
            self.current_frame = self.total_frames - 1
            return

        processing_time_ms = (time.time() - start_time) * 1000
        target_delay = (1000 / self.base_fps) / self.speed_multiplier
        actual_delay = max(1, int(target_delay - processing_time_ms))
        
        self._play_job = self.root.after(actual_delay,
                                         lambda: self.play_loop(generation))

    def on_closing(self):
        self._is_closing = True
        self._stop_playback()
            
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
                
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = CinePlayer(root)
    root.mainloop()
