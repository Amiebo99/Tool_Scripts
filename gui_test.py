
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Try to provide drag-and-drop if TkinterDnD2 is available
# pip install TkinterDnD2
DND_AVAILABE = False


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

    folder_segments = folder_path.split("/")
    print(folder_segments)

    # TODO: -------------------------------------------------------------
    # Insert your code right here. You have:
    #  - 'folder_path' (string)
    #  - 'folder_name' (string, just the final segment)
    #
    # Example (remove this and add your actual logic):
    # print(f"Processing folder: {folder_name} at {folder_path}")
    # -------------------------------------------------------------

    # For demo purposes, we’ll just return a message.
    return f"Ready to process '{folder_name}' at {folder_path}"


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


if __name__ == "__main__":
    main()
