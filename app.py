import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
import mysql.connector
from datetime import datetime
from io import BytesIO

# ------------------------- MySQL DB Credentials -------------------------
db_config = {
    'host': 'apiaov5.ccdeyfckavep.us-east-1.rds.amazonaws.com',
    'user': 'iaconsulta',
    'password': 'c1v2b3',
    'database': 'db_apiao',
    'port': 3306
}

# ------------------------- Load YOLOv8 model -------------------------
model = YOLO('yolov8_model/best.pt')

# ------------------------- Language selection -------------------------
language = st.sidebar.selectbox("🌐 Language / Idioma", ["Español", "English"])
lang = {
    "English": {
        "title": "🐚 Mussel Detector using YOLOv8",
        "subtitle": "Upload an image of mussels (JPG/PNG)",
        "upload": "Choose an image",
        "button": "🔍 Count Mussels in the Image",
        "save": "💾 Save to Database",
        "result_title": "✅ {} mussels detected",
        "big": "🟩 Big mussels: {} ({:.1f}%)",
        "small": "🟨 Small mussels: {} ({:.1f}%)",
        "label_big": "big",
        "label_small": "small",
    },
    "Español": {
        "title": "🐚 Contador de Semillas Standrews",
        "subtitle": "Sube una imagen de mejillones (JPG/PNG)",
        "upload": "Elige una imagen",
        "button": "🔍 Contar Mejillones en la Imagen",
        "save": "💾 Guardar en la Base de Datos",
        "result_title": "✅ {} mejillones detectados",
        "big": "🟩 Mejillones grandes: {} ({:.1f}%)",
        "small": "🟨 Mejillones pequeños: {} ({:.1f}%)",
        "label_big": "grande",
        "label_small": "pequeño",
    }
}[language]

st.set_page_config(page_title=lang["title"])
st.title(lang["title"])
st.subheader(lang["subtitle"])

# ------------------------- DB connection functions -------------------------
def get_db_connection():
    return mysql.connector.connect(**db_config)

def get_dropdown_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT id_area, nombre_area FROM areas")
    areas = cursor.fetchall()
    selected_area_id = st.selectbox("🗺️ Area", [a["id_area"] for a in areas], format_func=lambda x: next(a["nombre_area"] for a in areas if a["id_area"] == x))

    cursor.execute("SELECT id_centro, nombre_centro FROM centro_cultivo WHERE area_id = %s", (selected_area_id,))
    centros = cursor.fetchall()
    selected_centro_id = st.selectbox("🏗️ Centro de Cultivo", [c["id_centro"] for c in centros], format_func=lambda x: next(c["nombre_centro"] for c in centros if c["id_centro"] == x))

    cursor.execute("SELECT id_lineas, cod_lineas FROM lineas WHERE centros_id = %s", (selected_centro_id,))
    lineas = cursor.fetchall()
    selected_linea_id = st.selectbox("🧵 Línea", [l["id_lineas"] for l in lineas], format_func=lambda x: next(l["cod_lineas"] for l in lineas if l["id_lineas"] == x))

    cursor.execute("SELECT id_temporda, nombre FROM temporada")
    temporadas = cursor.fetchall()
    selected_temporada_id = st.selectbox("📅 Temporada", [t["id_temporda"] for t in temporadas], format_func=lambda x: next(t["nombre"] for t in temporadas if t["id_temporda"] == x))

    conn.close()
    return selected_area_id, selected_centro_id, selected_linea_id, selected_temporada_id

# ------------------------- Resize Image with Aspect Ratio -------------------------
def resize_with_aspect(image, target_size=(1200, 1600)):
    image_np = np.array(image.convert("RGB"))
    h, w = image_np.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (target_size[0] - nh) // 2
    bottom = target_size[0] - nh - top
    left = (target_size[1] - nw) // 2
    right = target_size[1] - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return padded, scale, left, top

# ------------------------- Upload Image -------------------------
uploaded_file = st.file_uploader(lang["upload"], type=["jpg", "jpeg", "png"])
MAX_FILE_SIZE_MB = 5

if uploaded_file:
    uploaded_file.seek(0, os.SEEK_END)
    file_size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large ({file_size_mb:.2f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB.")
    else:
        # Save image bytes to session
        st.session_state["image_bytes"] = uploaded_file.read()
        image = Image.open(BytesIO(st.session_state["image_bytes"]))
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button(lang["button"]):
            resized_img, scale, pad_x, pad_y = resize_with_aspect(image)
            results = model.predict(resized_img, imgsz=640)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            small_count = big_count = 0
            AREA_THRESHOLD = 2500
            sizes = []

            for (x1, y1, x2, y2) in boxes:
                area = (x2 - x1) * (y2 - y1)
                sizes.append(area)
                label = lang["label_big"] if area >= AREA_THRESHOLD else lang["label_small"]
                color = (0,255,0) if label == lang["label_big"] else (0,255,255)
                if area >= AREA_THRESHOLD:
                    big_count += 1
                else:
                    small_count += 1
                cv2.rectangle(resized_img, (x1, y1), (x2, y2), color, 2)

            total = big_count + small_count
            result_path = "resultado.jpg"
            cv2.imwrite(result_path, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

            # Save result in session
            st.session_state["result_image_path"] = result_path
            st.session_state["total"] = total
            st.session_state["big_count"] = big_count
            st.session_state["small_count"] = small_count
            st.session_state["sizes"] = sizes

# ------------------------- Show Results and Dropdowns (Even on Re-run) -------------------------
if "result_image_path" in st.session_state:
    st.image(st.session_state["result_image_path"], caption="Detection Result", use_container_width=True)
    st.success(lang["result_title"].format(st.session_state["total"]))
    st.write(lang["big"].format(st.session_state["big_count"], (st.session_state["big_count"] / st.session_state["total"] * 100)))
    st.write(lang["small"].format(st.session_state["small_count"], (st.session_state["small_count"] / st.session_state["total"] * 100)))

    st.markdown("---")
    st.subheader(lang["save"])

    id_area, id_centro, id_linea, id_temp = get_dropdown_data()

    st.markdown("### ✍️ Manual Entry (Optional)")
    manual_size = st.number_input("📏 Average Mussel Size (Manual)", min_value=0, value=0, step=1)
    manual_count = st.number_input("🔢 Mussel Count (Manual)", min_value=0, value=0, step=1)

    if st.button("✅ Save to Database"):
        conn = get_db_connection()
        cursor = conn.cursor()

        # Use manual values if entered, else use detected
        avg_area = manual_size if manual_size > 0 else int(np.mean(st.session_state["sizes"])) if st.session_state["sizes"] else 0
        total = manual_count if manual_count > 0 else st.session_state["total"]

        cursor.execute("""
            INSERT INTO reg_muestreo_siembra (id_area, id_centro, id_linea, id_temporada, cantidad, talla)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (id_area, id_centro, id_linea, id_temp, total, avg_area))

        conn.commit()
        conn.close()
        st.success("✅ Data saved to database!")

