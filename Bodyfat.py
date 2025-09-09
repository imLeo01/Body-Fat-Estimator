import os
import pickle
import traceback
import tkinter as tk
from tkinter import messagebox, filedialog, StringVar

import numpy as np
import pandas as pd
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History
import tensorflow as tf

from PIL import Image, ImageTk


FEATURES = [
    "Age", "Weight", "Height", "Neck", "Chest", "Abdomen",
    "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"
]

DEFAULT_MODEL_PATH = "bodyfat_ann_model.h5"
DEFAULT_SCALER_PATH = "bodyfat_scaler.pkl"


class BodyFatApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="cosmo")
        self.title("🏋️ Body Fat Estimator (ANN)")

        # full màn hình
        self.state("zoomed")

        self.model = None
        self.scaler = None
        self.model_path = None

        self._build_header()
        self._build_main_area()
        self._build_statusbar()

        self.after(200, self.try_autoload_defaults)

    # ================== UI ==================
    def _build_header(self):
        header = ttk.Frame(self, padding=15)
        header.pack(fill=X, pady=5)

        ttk.Label(
            header,
            text="Body Fat Estimator",
            font=("Helvetica", 32, "bold"),
            bootstyle=PRIMARY
        ).pack(side=LEFT, padx=20)

    def _build_main_area(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=BOTH, expand=True)

        # --- Left: Input form + buttons ---
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

        frame_inputs = ttk.Labelframe(
            left_frame, text="📏 Input Measurements",
            padding=20, bootstyle=INFO
        )
        frame_inputs.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.entries = {}
        half = (len(FEATURES) + 1) // 2

        for i, f in enumerate(FEATURES):
            col = 0 if i < half else 2
            row = i if i < half else i - half
            if f == "Age":
                unit = "year"
            elif f == "Weight":
                unit = "kg"
            else:
                unit = "cm"

           

            ttk.Label(frame_inputs, text=f"{f} ({unit})", font=("Arial", 12))\
                .grid(row=row, column=col, sticky="w", pady=6, padx=10)

            ent = ttk.Entry(frame_inputs, width=18, bootstyle=SUCCESS)
            ent.grid(row=row, column=col + 1, pady=6, padx=10)
            self.entries[f] = ent

        # Gender
        gender_row = ttk.Frame(frame_inputs)
        gender_row.grid(row=half, column=0, columnspan=2, pady=15, sticky="w")

        ttk.Label(gender_row, text="Gender:", font=("Arial", 12)).pack(side="left", padx=5)
        self.gender_var = StringVar(value="Male")
        for g in ["Male", "Female"]:
            ttk.Radiobutton(
                gender_row, text=g, variable=self.gender_var, value=g
            ).pack(side="left", padx=8)

        # Buttons dưới input
        btn_frame = ttk.Frame(left_frame, padding=10)
        btn_frame.pack(pady=5)

        btns = [
            ("📚 Train Model", PRIMARY, self.train_model),
            ("💾 Save Model", INFO, self.save_model),
            ("📂 Load Model", WARNING, self.load_model_dialog),
            ("🔮 Predict", SUCCESS, self.on_predict),
        ]

        for i, (text, style, cmd) in enumerate(btns):
            ttk.Button(
                btn_frame, text=text, bootstyle=style,
                command=cmd, width=20
            ).grid(row=0, column=i, padx=8, pady=8)

        # --- Right: Image ---
        try:
            img = Image.open("ANN/BTVN/fitness.jpg").resize((1200, 900))
            self.side_img = ImageTk.PhotoImage(img)
            img_label = ttk.Label(main_frame, image=self.side_img)
            img_label.pack(side=RIGHT, padx=20, pady=7)
        except Exception:
            pass

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="✅ Ready")
        ttk.Label(
            self, textvariable=self.status_var,
            bootstyle=SECONDARY, anchor="w", padding=6
        ).pack(side="bottom", fill="x")

    # ================== LOGIC ==================
    def try_autoload_defaults(self):
        try:
            if os.path.exists(DEFAULT_MODEL_PATH):
                if self._load_model(DEFAULT_MODEL_PATH):
                    self.status_var.set(f"Auto-loaded model: {DEFAULT_MODEL_PATH}")
                    self.model_path = DEFAULT_MODEL_PATH

            if os.path.exists(DEFAULT_SCALER_PATH) and self.scaler is None:
                with open(DEFAULT_SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                self.status_var.set(self.status_var.get() + " + scaler")
        except Exception:
            print("Auto-load defaults failed:", traceback.format_exc())

    # ---------------- TRAIN ----------------
    def train_model(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv")],
                title="Select training data (CSV)"
            )
            if not file_path:
                return

            df = pd.read_csv(file_path)

            # validate columns
            missing = [c for c in FEATURES + ["BodyFat"] if c not in df.columns]
            if missing:
                messagebox.showerror("Training Error", f"Dataset missing columns: {missing}")
                return

            X = df[FEATURES].values
            y = df["BodyFat"].values

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            self.model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='linear')
            ])
            self.model.compile(optimizer='adam', loss='mse')

            history = History()
            self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=80, batch_size=16,
                verbose=1, callbacks=[history]
            )

            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            self.status_var.set(f"Trained (test MSE={test_loss:.3f})")
            messagebox.showinfo("Training Complete", f"Model trained. Test MSE = {test_loss:.3f}")
            self.model_path = None  # mark unsaved
        except Exception as e:
            self.status_var.set("Training failed")
            messagebox.showerror("Training Error", f"{e}\n\n{traceback.format_exc()}")

    # ---------------- SAVE ----------------
    def save_model(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please train a model first before saving.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("Keras Model", "*.h5")],
            title="Save model as (.h5)"
        )
        if not file_path:
            return

        try:
            self.model.save(file_path)
            scaler_path = os.path.splitext(file_path)[0] + "_scaler.pkl"
            if self.scaler is not None:
                with open(scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)

            self.model_path = file_path
            self.status_var.set(f"Saved model: {file_path}")
            messagebox.showinfo(
                "Saved",
                f"Model saved to:\n{file_path}\nScaler saved to:\n{scaler_path}"
            )
        except Exception as e:
            messagebox.showerror("Save Error", f"{e}\n\n{traceback.format_exc()}")

    # ---------------- LOAD ----------------
    def load_model_dialog(self):
        path = filedialog.askopenfilename(
            filetypes=[("Keras Model", "*.h5 *.keras")],
            title="Select model file (.h5/.keras)"
        )
        if not path:
            return

        if not self._load_model(path):
            return

        # try load scaler
        scaler_path = os.path.splitext(path)[0] + "_scaler.pkl"
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.status_var.set(f"Model+scaler loaded: {os.path.basename(path)}")
                return
            except Exception:
                messagebox.showwarning("Scaler Load", "Model loaded but failed to load scaler.")

        # if scaler missing
        if messagebox.askyesno("Scaler not found", "No scaler file found next to model. Do you want to browse for a scaler (.pkl)?"):
            sp = filedialog.askopenfilename(
                filetypes=[("Pickle", "*.pkl"), ("All files", "*.*")],
                title="Select scaler (.pkl)"
            )
            if sp:
                try:
                    with open(sp, "rb") as f:
                        self.scaler = pickle.load(f)
                    self.status_var.set(f"Model loaded; scaler loaded: {os.path.basename(sp)}")
                    return
                except Exception:
                    messagebox.showwarning("Scaler Load", "Failed to load selected scaler file.")

        self.scaler = None
        self.status_var.set("Model loaded; scaler missing (predictions may be inaccurate)")

    def _load_model(self, path):
        """Robust model loader with fallback"""
        try:
            self.model = load_model(path, compile=False)
            self.model_path = path
            return True
        except Exception as e:
            try:
                custom_objects = {
                    'mse': tf.keras.losses.MeanSquaredError(),
                    'mae': tf.keras.losses.MeanAbsoluteError(),
                    'MeanSquaredError': tf.keras.losses.MeanSquaredError(),
                    'MeanAbsoluteError': tf.keras.losses.MeanAbsoluteError(),
                }
                self.model = load_model(path, custom_objects=custom_objects, compile=False)
                self.model_path = path
                return True
            except Exception as e2:
                messagebox.showerror("Load Error", f"Failed to load model:\n{e}\n\nFallback error:\n{e2}")
                return False

    # ---------------- CLASSIFY ----------------
    def classify_bodyfat(self, bf, gender):
        """Return bodyfat category"""
        if gender == "Male":
            if bf < 6: return "Essential Fat"
            elif bf < 14: return "Athletes"
            elif bf < 18: return "Fitness"
            elif bf < 25: return "Average"
            else: return "Obese"
        else:
            if bf < 14: return "Essential Fat"
            elif bf < 21: return "Athletes"
            elif bf < 25: return "Fitness"
            elif bf < 32: return "Average"
            else: return "Obese"

    # ---------------- PREDICT ----------------
    def on_predict(self):
        try:
            if self.model is None:
                raise ValueError("No model loaded. Train or load a model first.")
            if self.scaler is None:
                raise ValueError("Scaler missing. Load scaler file or train model in this session.")

            vals = []
            for f in FEATURES:
                txt = self.entries[f].get().strip()
                if not txt:
                    raise ValueError(f"Missing {f}")
                vals.append(float(txt))

            # Convert units: Weight (kg→lbs), Height (cm→inches)
            vals[1] *= 2.20462
            vals[2] /= 2.54

            X_input = np.array(vals).reshape(1, -1)
            X_scaled = self.scaler.transform(X_input)
            pred = float(self.model.predict(X_scaled, verbose=0)[0, 0])

            gender = self.gender_var.get()
            category = self.classify_bodyfat(pred, gender)

            messagebox.showinfo(
                "Prediction",
                f"Estimated Body Fat: {pred:.2f}%\nGender: {gender}\nCategory: {category}"
            )
            self.status_var.set(f"Last prediction: {pred:.2f}% ({category})")

        except Exception as e:
            msg = str(e)
            if "No model loaded" in msg:
                msg += ("\n\nHint: Train a model (Train Model) then Save Model, "
                        "or use Load Model to load a previously saved .h5 file.")
            if "Scaler missing" in msg:
                msg += ("\n\nHint: Make sure the scaler (.pkl) saved alongside the model "
                        "is present, or train in this session.")
            messagebox.showerror("Prediction Error", msg)


# ================== RUN ==================
if __name__ == "__main__":
    app = BodyFatApp()
    app.mainloop()
