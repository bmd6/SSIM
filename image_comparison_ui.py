import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import cv2

# Import the core logic functions from your file
from image_comparison_logic import calculate_ssim_full_image, calculate_ssim_region

class ImageComparisonApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Similarity Comparator (SSIM)")
        master.geometry("1200x900")

        # --- State and Data Variables ---
        self.img1_path = tk.StringVar()
        self.img2_path = tk.StringVar()
        self.comparison_type = tk.StringVar(value="full")
        self.select_on_image = tk.StringVar(value="1")

        # Region coordinates
        self.region_x = tk.StringVar(value="0")
        self.region_y = tk.StringVar(value="0")
        self.region_w = tk.StringVar(value="100")
        self.region_h = tk.StringVar(value="100")

        # --- Internal State ---
        self.selection_active = False
        self.start_x = None
        self.start_y = None
        self.current_drawing_rect = None
        self.persistent_rect_coords = None
        self.image_for_selection = None

        self.setup_ui()

    def setup_ui(self):
        # --- Input Frame ---
        input_frame = ttk.LabelFrame(self.master, text="Image Selection and Comparison Type")
        input_frame.pack(padx=10, pady=10, fill="x", expand=False)

        # Image 1
        ttk.Label(input_frame, text="Image 1 Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.img1_path, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=lambda: self.browse_image(self.img1_path)).grid(row=0, column=2, padx=5, pady=5)

        # Image 2
        ttk.Label(input_frame, text="Image 2 Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(input_frame, textvariable=self.img2_path, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=lambda: self.browse_image(self.img2_path)).grid(row=1, column=2, padx=5, pady=5)

        # Comparison Type Radio Buttons
        ttk.Radiobutton(input_frame, text="Full Image Comparison", variable=self.comparison_type,
                        value="full", command=self.toggle_region_inputs).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(input_frame, text="Compare Specific Region", variable=self.comparison_type,
                        value="region", command=self.toggle_region_inputs).grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Region Input Frame
        self.region_frame = ttk.LabelFrame(input_frame, text="Region Coordinates (x, y, width, height)")
        self.region_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        self.region_inputs = []
        labels = ["X:", "Y:", "Width:", "Height:"]
        vars_list = [self.region_x, self.region_y, self.region_w, self.region_h]
        for i, (label_text, var) in enumerate(zip(labels, vars_list)):
            ttk.Label(self.region_frame, text=label_text).grid(row=0, column=i*2, padx=2, pady=2, sticky="e")
            entry = ttk.Entry(self.region_frame, textvariable=var, width=8)
            entry.grid(row=0, column=i*2+1, padx=2, pady=2, sticky="w")
            self.region_inputs.append(entry)

        ttk.Label(self.region_frame, text="Select on:").grid(row=0, column=8, padx=5, pady=5, sticky="e")
        self.rb_img1 = ttk.Radiobutton(self.region_frame, text="Image 1", variable=self.select_on_image, value="1")
        self.rb_img1.grid(row=0, column=9, padx=2, pady=2, sticky="w")
        self.rb_img2 = ttk.Radiobutton(self.region_frame, text="Image 2", variable=self.select_on_image, value="2")
        self.rb_img2.grid(row=0, column=10, padx=2, pady=2, sticky="w")

        self.select_region_button = ttk.Button(self.region_frame, text="Draw Region", command=self.start_region_selection)
        self.select_region_button.grid(row=1, column=8, columnspan=3, padx=5, pady=5, sticky="ew")

        self.toggle_region_inputs()

        # Action buttons
        action_frame = ttk.Frame(input_frame)
        action_frame.grid(row=5, column=1, pady=10)

        ttk.Button(action_frame, text="Compare Images", command=self.compare_images_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Exit", command=self.master.destroy).pack(side=tk.LEFT, padx=5)

        # --- Results Frame ---
        results_frame = ttk.LabelFrame(self.master, text="Comparison Results")
        results_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.ssim_score_label = ttk.Label(results_frame, text="SSIM Score: N/A", font=("Arial", 12, "bold"))
        self.ssim_score_label.pack(pady=5)

        self.fig = plt.Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, results_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X, expand=False)

    def browse_image(self, path_var):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            path_var.set(file_path)
            self.redisplay_current_images()

    def toggle_region_inputs(self):
        state = "normal" if self.comparison_type.get() == "region" else "disabled"
        for widget in self.region_inputs:
            widget.config(state=state)
        self.select_region_button.config(state=state)
        self.rb_img1.config(state=state)
        self.rb_img2.config(state=state)
        self.redisplay_current_images()

    def clear_plot(self):
        self.ax.clear()
        self.ax.axis('off')
        self.canvas.draw_idle()

    def embed_plot(self, new_fig):
        parent_frame = self.ssim_score_label.master

        if self.canvas_widget: self.canvas_widget.destroy()
        if self.toolbar: self.toolbar.destroy()

        self.fig = new_fig
        self.ax = self.fig.get_axes()[0] if self.fig.get_axes() else self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, parent_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X, expand=False)
        self.canvas.draw()

    def start_region_selection(self):
        selected_image_num = self.select_on_image.get()
        img_path = self.img1_path.get() if selected_image_num == "1" else self.img2_path.get()
        display_title = f"Draw Region on Image {selected_image_num}"

        if not img_path:
            messagebox.showerror("Error", f"Please select Image {selected_image_num} first.")
            return

        try:
            self.image_for_selection = cv2.imread(img_path)
            if self.image_for_selection is None:
                raise IOError(f"Could not load Image {selected_image_num}.")

            self.selection_active = True
            self.master.config(cursor="crosshair")
            messagebox.showinfo("Region Selection", f"Click and drag on {display_title} to select a region.\nRight-click or press Esc to cancel.")

            self.clear_plot()
            self.ax.imshow(cv2.cvtColor(self.image_for_selection, cv2.COLOR_BGR2RGB))
            self.ax.set_title(display_title)
            self.ax.axis('on')
            self.fig.tight_layout()
            self.canvas.draw_idle()

            self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
            self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
            self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
            self.cid_key = self.canvas.mpl_connect('key_press_event', self.on_key_press)

        except Exception as e:
            messagebox.showerror("Error", f"Could not prepare image for selection: {e}")
            self.end_region_selection(success=False)

    def end_region_selection(self, success=True):
        self.selection_active = False
        self.master.config(cursor="")
        self.start_x = self.start_y = None

        if self.current_drawing_rect and self.current_drawing_rect in self.ax.patches:
            self.current_drawing_rect.remove()
        self.current_drawing_rect = None

        if hasattr(self, 'cid_press'): self.canvas.mpl_disconnect(self.cid_press)
        if hasattr(self, 'cid_motion'): self.canvas.mpl_disconnect(self.cid_motion)
        if hasattr(self, 'cid_release'): self.canvas.mpl_disconnect(self.cid_release)
        if hasattr(self, 'cid_key'): self.canvas.mpl_disconnect(self.cid_key)

        if success and self.persistent_rect_coords:
            self.clear_plot()
            self.ax.imshow(cv2.cvtColor(self.image_for_selection, cv2.COLOR_BGR2RGB))
            x, y, w, h = self.persistent_rect_coords
            final_rect = plt.Rectangle((x, y), w, h, edgecolor='lime', facecolor='none', lw=2, linestyle='--')
            self.ax.add_patch(final_rect)
            self.ax.set_title(f"Selected Region ({x},{y},{w},{h})")
            self.ax.axis('on')
            self.fig.tight_layout()
            self.canvas.draw_idle()
        else:
            self.redisplay_current_images()

        self.image_for_selection = None
        if not success: self.ssim_score_label.config(text="SSIM Score: N/A")

    def on_press(self, event):
        if not (self.selection_active and event.inaxes == self.ax and event.button == 1): return
        self.start_x, self.start_y = int(event.xdata), int(event.ydata)
        if self.current_drawing_rect: self.current_drawing_rect.remove()
        self.current_drawing_rect = plt.Rectangle((self.start_x, self.start_y), 0, 0, edgecolor='red', facecolor='none', lw=2)
        self.ax.add_patch(self.current_drawing_rect)
        self.canvas.draw_idle()

    def on_motion(self, event):
        if not (self.selection_active and self.start_x is not None and event.inaxes == self.ax and event.button == 1): return
        cur_x, cur_y = int(event.xdata), int(event.ydata)
        width, height = abs(cur_x - self.start_x), abs(cur_y - self.start_y)
        x, y = min(self.start_x, cur_x), min(self.start_y, cur_y)
        self.current_drawing_rect.set_xy((x, y))
        self.current_drawing_rect.set_width(width)
        self.current_drawing_rect.set_height(height)
        self.canvas.draw_idle()

    def on_release(self, event):
        if self.selection_active and event.button == 3:
            messagebox.showinfo("Selection Cancelled", "Region selection cancelled.")
            self.end_region_selection(success=False)
            return

        if not (self.selection_active and self.start_x is not None and event.inaxes == self.ax and event.button == 1): return
        end_x, end_y = int(event.xdata), int(event.ydata)
        x, y = min(self.start_x, end_x), min(self.start_y, end_y)
        width, height = abs(end_x - self.start_x), abs(end_y - self.start_y)

        if width < 1 or height < 1:
            messagebox.showwarning("Selection", "Selected region is too small. Please try again.")
            self.end_region_selection(success=False)
            return

        self.region_x.set(str(x))
        self.region_y.set(str(y))
        self.region_w.set(str(width))
        self.region_h.set(str(height))
        self.persistent_rect_coords = (x, y, width, height)
        self.end_region_selection(success=True)

    def on_key_press(self, event):
        if self.selection_active and event.key == 'escape':
            messagebox.showinfo("Selection Cancelled", "Region selection cancelled.")
            self.end_region_selection(success=False)

    def compare_images_action(self):
        if self.selection_active:
            messagebox.showwarning("Active Selection", "Please finish or cancel the region selection first.")
            return

        img1_path, img2_path = self.img1_path.get(), self.img2_path.get()
        if not img1_path or not img2_path:
            messagebox.showerror("Error", "Please select both image files.")
            return

        self.persistent_rect_coords = None
        score, fig_to_display = None, None

        try:
            if self.comparison_type.get() == "full":
                score, fig_to_display = calculate_ssim_full_image(img1_path, img2_path)
            else:
                x, y, w, h = int(self.region_x.get()), int(self.region_y.get()), int(self.region_w.get()), int(self.region_h.get())
                if w <= 0 or h <= 0:
                    raise ValueError("Width and Height must be positive integers.")
                score, fig_to_display = calculate_ssim_region(img1_path, img2_path, (x, y, w, h))
        except Exception as e:
            messagebox.showerror("Comparison Error", f"An error occurred: {e}")
            self.redisplay_current_images()
            return

        if score is not None and fig_to_display is not None:
            self.ssim_score_label.config(text=f"SSIM Score: {score:.4f}")
            self.embed_plot(fig_to_display)
        else:
            self.ssim_score_label.config(text="SSIM Score: Error during calculation.")
            self.redisplay_current_images()

    def redisplay_current_images(self):
        if not hasattr(self, 'fig') or not self.fig: return

        self.fig.clear()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        self.ax = ax1 # Reassign main axis reference

        self.ssim_score_label.config(text="SSIM Score: N/A")

        def display_image_on_ax(ax_obj, img_path, title):
            ax_obj.set_title(title)
            ax_obj.axis('off')
            if img_path:
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        ax_obj.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else:
                        ax_obj.text(0.5, 0.5, "Image not loaded", ha='center', va='center', transform=ax_obj.transAxes)
                except Exception:
                    ax_obj.text(0.5, 0.5, "Error loading image", ha='center', va='center', transform=ax_obj.transAxes)
            else:
                ax_obj.text(0.5, 0.5, "No Image Selected", ha='center', va='center', transform=ax_obj.transAxes)

        display_image_on_ax(ax1, self.img1_path.get(), "Image 1")
        display_image_on_ax(ax2, self.img2_path.get(), "Image 2")

        self.fig.tight_layout()
        self.canvas.draw_idle()

def main():
    root = tk.Tk()
    app = ImageComparisonApp(root)
    app.redisplay_current_images()
    root.mainloop()

if __name__ == "__main__":
    main()