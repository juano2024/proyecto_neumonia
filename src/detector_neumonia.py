#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
detector_neumonia.py - Version depurada
Herramienta de apoyo diagnostico de neumonia con Grad-CAM.
"""

import csv
import numpy as np
import cv2
import pydicom as dicom
import tensorflow as tf
from tensorflow.keras.models import load_model

from tkinter import Tk, StringVar, Text, END
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image


MODEL_PATH = "conv_MLP_84.h5"
_model_cache = None


def model_fun():
    """Carga (una sola vez) y devuelve el modelo entrenado."""
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model(MODEL_PATH, compile=False)
    return _model_cache


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255.0
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array


def grad_cam(array):
    img = preprocess(array)
    model = model_fun()
    last_conv_layer = model.get_layer("conv10_thisone")
    grad_model = tf.keras.models.Model(
        model.inputs, [last_conv_layer.output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img)
        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(conv_output, list):
            conv_output = conv_output[0]
        argmax = tf.argmax(preds[0])
        loss = preds[:, argmax]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img2 = cv2.resize(array, (512, 512))
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    transparency = (heatmap * 0.8).astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2).astype(np.uint8)
    return superimposed_img[:, :, ::-1]


def predict(array):
    batch_array_img = preprocess(array)
    model = model_fun()
    preds = model.predict(batch_array_img)
    prediction = np.argmax(preds)
    proba = np.max(preds) * 100

    labels = {0: "bacteriana", 1: "normal", 2: "viral"}
    label = labels.get(prediction, "desconocido")

    heatmap = grad_cam(array)
    return label, proba, heatmap


def read_dicom_file(path):
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    max_val = img2.max() if img2.max() != 0 else 1
    img2 = (np.maximum(img2, 0) / max_val) * 255.0
    img2 = np.uint8(img2)
    img_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_rgb, img2show


def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    img2 = img_array.astype(float)
    max_val = img2.max() if img2.max() != 0 else 1
    img2 = (np.maximum(img2, 0) / max_val) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS  # compatibilidad Pillow>=10


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la deteccion rapida de neumonia")
        fonti = font.Font(weight="bold")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        self.lab1 = ttk.Label(self.root, text="Imagen Radiografica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cedula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNOSTICO MEDICO DE NEUMONIA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        self.ID = StringVar()
        self.result = StringVar()

        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.text1.focus_set()

        self.array = None
        self.img1 = None
        self.img2 = None
        self.label = ""
        self.proba = 0.0
        self.reportID = 0

        self.root.mainloop()

    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if not filepath:
            return

        # Antes siempre se leia como DICOM sin importar la extension
        ext = filepath.lower().rsplit(".", 1)[-1]
        if ext == "dcm":
            self.array, img2show = read_dicom_file(filepath)
        else:
            self.array, img2show = read_jpg_file(filepath)

        self.img1 = img2show.resize((250, 250), RESAMPLE)
        self.img1 = ImageTk.PhotoImage(self.img1)
        self.text_img1.image_create(END, image=self.img1)
        self.button1["state"] = "enabled"

    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), RESAMPLE)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a", newline="") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
        showinfo(title="Guardar", message="Los datos se guardaron con exito.")

    def create_pdf(self):
        import tkcap
        cap = tkcap.CAP(self.root)
        img_path = "Reporte" + str(self.reportID) + ".jpg"
        cap.capture(img_path)
        img = Image.open(img_path)
        img = img.convert("RGB")
        pdf_path = "Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con exito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmacion", message="Se borraran todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            # Text.delete espera indices, no objetos de imagen
            self.text_img1.delete("1.0", "end")
            self.text_img2.delete("1.0", "end")
            showinfo(title="Borrar", message="Los datos se borraron con exito")


def main():
    App()
    return 0


if __name__ == "__main__":
    main()