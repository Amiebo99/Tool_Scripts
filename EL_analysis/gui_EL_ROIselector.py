# file for reading in images and counting pixels above a threshold

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import csv
from typing import Callable, Dict, Iterable, Optional, Union
import sys


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Try to provide drag-and-drop if TkinterDnD2 is available
# pip install TkinterDnD2
DND_AVAILABLE = False

class FolderDropboxApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Folder Dropbox")

        # Main frame
        container = ttk.Frame(root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        # Instruction
        ttk.Label(
            container,
            text="Drop a folder here (if enabled) or choose a folder to process.",
        ).pack(anchor="w")

        # Drop area or fallback
        if DND_AVAILABLE and isinstance(root, TkinterDnD.Tk):
            self.drop_area = ttk.Frame(container, relief=tk.RIDGE, padding=24)
            self.drop_area.pack(fill=tk.BOTH, expand=True, pady=8)
            self.drop_label = ttk.Label(
                self.drop_area,
                text="📦 Drop Folder Here",
                anchor="center",
                padding=8,
            )
            self.drop_label.pack(fill=tk.BOTH, expand=True)

            # Register as drop target
            self.drop_area.drop_target_register(DND_FILES)
            self.drop_area.dnd_bind("<<Drop>>", self.on_drop)
        else:
            # Fallback message when DND isn’t available
            self.drop_area = ttk.Frame(container, relief=tk.RIDGE, padding=24)
            self.drop_area.pack(fill=tk.BOTH, expand=True, pady=8)
            ttk.Label(
                self.drop_area,
                text="Drag-and-drop not available.\nUse 'Choose Folder…' below.",
                anchor="center",
                padding=8,
            ).pack(fill=tk.BOTH, expand=True)

        # Selected path display
        path_frame = ttk.Frame(container)
        path_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(path_frame, text="Selected folder:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value="")
        self.path_entry = ttk.Entry(path_frame, textvariable=self.path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        # Buttons
        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=tk.X, pady=8)

        choose_btn = ttk.Button(btn_frame, text="Choose Folder…", command=self.choose_folder)
        choose_btn.pack(side=tk.LEFT)

        process_btn = ttk.Button(btn_frame, text="Process", command=self.run_process)
        process_btn.pack(side=tk.RIGHT)

        # Status area
        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(container, textvariable=self.status_var, foreground="#555")
        status.pack(fill=tk.X, pady=(8, 0))

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select folder to process")
        if folder:
            self.path_var.set(folder)
            self.status_var.set("Folder selected. Click Process to continue.")

    def on_drop(self, event):
        """
        Handle drag-and-drop. TkinterDnD passes a space-separated list of paths,
        potentially wrapped in braces. We’ll grab the first path and clean it.
        """
        raw = event.data.strip()
        # Remove curly braces around paths that contain spaces
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]

        # If multiple paths were dropped, take the first
        first_path = raw.split("} {")[0].split()  # handle variations
        candidate = first_path[0] if isinstance(first_path, list) else raw

        p = Path(candidate)
        if p.is_dir():
            self.path_var.set(str(p.resolve()))
            self.status_var.set("Folder dropped. Click Process to continue.")
        else:
            messagebox.showwarning("Not a folder", "Please drop a folder (not a file).")

    def run_process(self):
        folder_path = self.path_var.get().strip()
        if not folder_path:
            messagebox.showinfo("No folder", "Please choose or drop a folder first.")
            return

        p = Path(folder_path)
        if not p.exists() or not p.is_dir():
            messagebox.showerror("Invalid folder", "The selected path is not a valid folder.")
            return

        try:
            # Call your processing function
            msg = process_folder(folder_path)
            self.status_var.set(msg or "Processing complete.")
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{e}")
            self.status_var.set("Error occurred.")


def main():
    # Create root. Use TkinterDnD.Tk if available to enable drag-and-drop.
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    # Light styling
    try:
        # Use the platform's native theme if possible
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = FolderDropboxApp(root)
    root.geometry("560x280")
    root.mainloop()



def count_white_pixels(image: np.ndarray) -> int:
    """
    Counts the number of pixels in a given OpenCV image that have a value of 255.

    Works correctly for both grayscale and color (BGR) images.

    Args:
        image: A NumPy array representing the OpenCV image (BGR or Grayscale).

    Returns:
        The total count of pixels with a value of 255.
    """
    if image is None:
        print("Error: Input image is None.")
        return 0

    # If the image is color (BGR), convert it to grayscale first
    # White in BGR is (255, 255, 255), which becomes 255 in grayscale
    if len(image.shape) == 3:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        gray_image = image

    # Use NumPy's count_nonzero function on a boolean mask
    # The mask checks if each pixel in the grayscale image is exactly 255
    white_pixel_count = np.count_nonzero(gray_image == 255)

    return int(white_pixel_count)

#perform thresholds
def perform_threshold(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.normalize(img, None, 8, 32, cv.NORM_MINMAX)
    
    # global thresholding
    ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    
    # Otsu's thresholding
    ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    # triangle thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (1,1), 0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_TRIANGLE)
    
    # plot all the images and their histograms
    images = [img, 0, th1,
            img, 0, th2,
            blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
            'Original Noisy Image','Histogram',"Otsu's Thresholding",
            'Gaussian filtered Image','Histogram',"Triangle Thresholding"]
    
    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()
    print(ret3)
    return th3


def select_scaled_roi(image_path, screen_max_width=1200, screen_max_height=800):
    # 1. Load the original image
    img_original = cv.imread(image_path)
    ## normalising brightness in the image before scaling and/or selecting roi
    if img_original is None:
        print(f"Error: Could not load image from {image_path}")
        return

    original_height, original_width = img_original.shape[:2]

    # 2. Calculate scaling factor to fit screen while maintaining aspect ratio
    # Calculate potential scale based on max width/height
    scale_width = screen_max_width / original_width
    scale_height = screen_max_height / original_height
    # Use the smaller scale factor to ensure the entire image fits within screen limits
    scale_factor = min(scale_width, scale_height)

    # If the image is already small enough, do not scale up
    if scale_factor >= 1:
        scale_factor = 1.0
        img_display = img_original.copy()
    else:
        # Resize the image for display
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        dim = (new_width, new_height)
        # Use INTER_AREA for downsampling for better quality
        img_display = cv.resize(img_original, dim, interpolation=cv.INTER_AREA)

    # 3. Select ROI on the *resized* display image
    print("Drag a rectangle with the mouse over the display image.")
    print("Press ENTER or SPACE to confirm the selection, or 'c' to cancel.")
    r_display = cv.selectROI("Select ROI on Scaled Image", img_display, showCrosshair=True, fromCenter=False)
    cv.destroyWindow("Select ROI on Scaled Image")

    x_display, y_display, w_display, h_display = r_display

    if w_display > 0 and h_display > 0:
        # 4. Scale coordinates back to the *original* image size
        x_original = int(x_display / scale_factor)
        y_original = int(y_display / scale_factor)
        w_original = int(w_display / scale_factor)
        h_original = int(h_display / scale_factor)

        # 5. Crop and process the *original* image
        cropped_original = img_original[y_original:y_original+h_original, x_original:x_original+w_original]

        # *** Do something to the selected area (example: apply a blur) ***
        processed_roi_original = cropped_original
        # Display results (can resize the original result for display purposes if still too large)
        #cv.imshow("Original Image (Full)", img_original)
        #cv.imshow("Processed ROI (Blurred)", processed_roi_original)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Selection canceled or invalid selection.")
    

    return processed_roi_original


from pathlib import Path
import csv
from typing import Callable, Union


def process_jpgs_to_csv_int(
    folder: Union[str, Path],
    csv_path: Union[str, Path],
    operation_fn: Callable[[Path], int],
    recursive: bool = False,
    filename_field: str = "filename",
    value_field: str = "value",
    encoding: str = "utf-8",
) -> None:
    """
    Iterate over .jpg files in a folder, run an operation that returns a single int,
    and write (create or rewrite) a CSV with rows [filename, value].

    Parameters
    ----------
    folder : str | Path
        Folder to scan for .jpg files.
    csv_path : str | Path
        Output CSV file (will be created/re-written).
    operation_fn : Callable[[Path], int]
        Function that accepts an image Path and returns an integer result.
    recursive : bool, default False
        Search subfolders if True.
    filename_field : str, default "filename"
        Column name for the image filename.
    value_field : str, default "value"
        Column name for the integer result.
    encoding : str, default "utf-8"
        CSV encoding.
    """
    folder = Path(folder)
    csv_path = Path(csv_path)

    pattern = "**/*.jpg" if recursive else "*.jpg"
    jpg_files = sorted(folder.glob(pattern))

    if not jpg_files:
        print(f"[INFO] No .jpg files found in {folder} (recursive={recursive}).")
        return

    rows = []
    for img_path in jpg_files:
        try:
            result_int = operation_fn(img_path)
            if not isinstance(result_int, int):
                raise TypeError("operation_fn must return an int.")
            print(img_path.name.split())
            rows.append({filename_field: img_path.name, value_field: result_int})
        except Exception as e:
            # Log error and continue
            print(f"[WARN] Failed to process {img_path}: {e}")

    if not rows:
        print("[INFO] No rows produced; CSV will not be written.")
        return

    # Write CSV (overwrite or create)
    with csv_path.open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=[filename_field, value_field])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {csv_path}.")




# Load the image
#img = select_scaled_roi('2025_10_03/OK/250626 200TC_20251003.jpg') # Replace "your_image.jpg" with your image file


# Display the image and allow the user to select an ROI
# The 'False' argument means the crosshair will not be shown initially
def process_image(nameOfFile: Path) -> int:
    roi = select_scaled_roi(nameOfFile)


    processed_roi = perform_threshold(roi)

    # Display the original image, the cropped ROI, and the processed ROI
    #cv.imshow("Original Image", roi)
    #cv.imshow("Cropped ROI", roi)
    #cv.imshow("Processed ROI (Grayscale)", processed_roi)
    # 3. Call the function
    count = count_white_pixels(processed_roi)

    print(f"Total number of 255-value pixels: {count}")

    cv.waitKey(0) # Wait indefinitely for a key press
    cv.destroyAllWindows() # Close all OpenCV windows
    return count
def process_folder(folder_path: str):
    """
    PLACEHOLDER: Insert your processing code here.

    Parameters
    ----------
    folder_path : str
        Absolute path to the selected/dropped folder.

    You can compute whatever you need here. Example:
    - Iterate files
    - Run your operation
    - Write outputs (CSV, images, etc.)

    Example scaffolding (commented out):
    ------------------------------------
    from pathlib import Path
    p = Path(folder_path)
    folder_name = p.name  # <-- captured folder name
    # TODO: do something with folder_name and contents
    # for jpg in p.glob("*.jpg"):
    #     result_int = your_operation(jpg)
    #     # collect results / write to CSV / etc.
    """
    p = Path(folder_path)

    # Capture the folder name for your use
    folder_name = p.name

    #folder_segments = folder_path.split("/")
    #print(folder_segments)

    ## running the functions
    date = p.name
    folderName = date + '/OK/'
    outputFilename = date + '_equalised_testing.csv'
    process_jpgs_to_csv_int(folderName, outputFilename, process_image, recursive=False)

    # For demo purposes, we’ll just return a message.
    return f"processed'{folder_name}' at {folder_path}"


if __name__ == "__main__":
    main()
