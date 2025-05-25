import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

# --- Import hàm infer ---
try:
    from Infer import infer_from_row  # type: ignore
except ImportError:
    # Khi Infer.py chưa nằm trong PYTHONPATH ➜ import động
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "Infer", os.path.join(os.path.dirname(__file__), "Infer.py")
    )
    Infer = importlib.util.module_from_spec(spec)
    sys.modules["Infer"] = Infer  # type: ignore
    spec.loader.exec_module(Infer)  # type: ignore
    infer_from_row = Infer.infer_from_row  # type: ignore


class DiagnoseApp:
    """Tkinter GUI"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CSV Chẩn Đoán GUI")
        self.csv_path: str | None = None
        self.df: pd.DataFrame | None = None

        # --- Thanh công cụ ---
        toolbar = ttk.Frame(root, padding=4)
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="📂 Mở CSV", command=self.open_csv).pack(
            side="left", padx=2
        )
        self.diagnose_btn = ttk.Button(
            toolbar,
            text="🔬 Chẩn đoán dòng chọn",
            command=self.diagnose_selected,
            state="disabled",
        )
        self.diagnose_btn.pack(side="left", padx=2)

        # --- Bảng dữ liệu ---
        self.tree_frame = ttk.Frame(root)
        self.tree_frame.pack(fill="both", expand=True)
        self.tree: ttk.Treeview | None = None

        # --- Thanh trạng thái ---
        self.status_var = tk.StringVar(value="Chưa mở file CSV")
        ttk.Label(
            root,
            textvariable=self.status_var,
            padding=4,
            relief="sunken",
            anchor="w",
        ).pack(fill="x")

    # ---------------------------------------------------------------------
    #  Chọn & load CSV
    # ---------------------------------------------------------------------
    def open_csv(self):
        path = filedialog.askopenfilename(
            title="Chọn file CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Lỗi đọc file", f"Không thể đọc file CSV:\n{exc}")
            return

        # Đảm bảo có cột "Chẩn Đoán"
        if "Chẩn Đoán" not in df.columns:
            df["Chẩn Đoán"] = ""

        self.csv_path = path
        self.df = df
        self._build_table()
        self.status_var.set(
            f"Đã mở: {os.path.basename(path)} | {len(df)} dòng"
        )
        self.diagnose_btn["state"] = "normal"

    # ------------------------------------------------------------------
    #  Tạo Treeview hiển thị DataFrame
    # ------------------------------------------------------------------
    def _build_table(self):
        # Xóa "tree" cũ (nếu có)
        for child in self.tree_frame.winfo_children():
            child.destroy()

        cols = list(self.df.columns)  # type: ignore[attr-defined]
        self.tree = ttk.Treeview(self.tree_frame, columns=cols, show="headings")

        vsb = ttk.Scrollbar(
            self.tree_frame, orient="vertical", command=self.tree.yview
        )
        hsb = ttk.Scrollbar(
            self.tree_frame, orient="horizontal", command=self.tree.xview
        )
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.tree_frame.rowconfigure(0, weight=1)
        self.tree_frame.columnconfigure(0, weight=1)

        # Đặt tiêu đề cột & width cơ bản
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

        # Đổ dữ liệu
        for idx, row in self.df.iterrows():  # type: ignore[attr-defined]
            self.tree.insert("", "end", iid=str(idx), values=row.tolist())

    # ------------------------------------------------------------------
    #  Chẩn đoán dòng được chọn
    # ------------------------------------------------------------------
    def diagnose_selected(self):
        if not self.tree or self.df is None or self.csv_path is None:
            return
        item_id = self.tree.focus()
        if not item_id:
            messagebox.showwarning("Chưa chọn dòng", "Vui lòng chọn một dòng để chẩn đoán.")
            return
        row_idx = int(item_id)

        def _worker():
            try:
                self.status_var.set(f"Đang chẩn đoán dòng {row_idx}...")
                result = infer_from_row(self.csv_path, row_idx)
                # Cập nhật DataFrame & UI
                self.df.at[row_idx, "Chẩn Đoán"] = result  # type: ignore[index]
                self.tree.set(item_id, column="Chẩn Đoán", value=str(result))
                messagebox.showinfo("Kết quả", f"Dòng {row_idx}: {result}")
            except Exception as exc:
                messagebox.showerror("Lỗi Infer", str(exc))
            finally:
                self.status_var.set("Hoàn tất")

        # Chạy infer trong thread riêng để tránh đơ UI
        threading.Thread(target=_worker, daemon=True).start()


# ---------------------------------------------------------------------
#  Chạy ứng dụng
# ---------------------------------------------------------------------
if __name__ == "__main__":
    root_tk = tk.Tk()
    app = DiagnoseApp(root_tk)
    root_tk.mainloop()
