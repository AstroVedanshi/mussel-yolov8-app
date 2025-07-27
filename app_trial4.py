import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import os
import cv2
import numpy as np
#####################################################################################################
import mysql.connector
from mysql.connector import Error

def connect_to_db():
    try:
        conn = mysql.connector.connect(
            host='apiaov5.ccdeyfckavep.us-east-1.rds.amazonaws.com',
            user='iaconsulta',
            password='c1v2b3',
            database='apiaov5',
            port=3306
        )
        return conn
    except Error as e:
        st.error(f"❌ Could not connect to the database: {e}")
        return None
###########################################################################333        
# Load YOLOv8 model
model = YOLO('yolov8_model/best.pt')

# Language selection
language = st.sidebar.selectbox("🌐 Language / Idioma", ["Español", "English"])
lang = {
    "English": {
        "title": "🐚 Mussel Detector using YOLOv8",
        "subtitle": "Upload an image of mussels (JPG/PNG)",
        "upload": "Choose an image",
        "button": "🔍 Count Mussels in the Image",
        "error": "🚫 File too large: {:.2f} MB. Max allowed is {} MB.",
        "spinner": "Detecting mussels...",
        "image_caption": "Uploaded Image",
        "label_big": "big",
        "label_small": "small",
        "result_title": "✅ {} mussels detected",
        "big": "🟩 Big mussels: {} ({:.1f}%)",
        "small": "🟨 Small mussels: {} ({:.1f}%)",
        "result_image": "Detection Result",
    },
    "Español": {
        "title": "🐚 Contador de Semillas Standrews",
        "subtitle": "Sube una imagen de mejillones (JPG/PNG)",
        "upload": "Elige una imagen",
        "button": "🔍 Contar Mejillones en la Imagen",
        "error": "🚫 Archivo demasiado grande: {:.2f} MB. El máximo permitido es {} MB.",
        "spinner": "Detectando mejillones...",
        "image_caption": "Imagen subida",
        "label_big": "grande",
        "label_small": "pequeño",
        "result_title": "✅ {} mejillones detectados",
        "big": "🟩 Mejillones grandes: {} ({:.1f}%)",
        "small": "🟨 Mejillones pequeños: {} ({:.1f}%)",
        "result_image": "Resultado de la Detección",
    }
}[language]

# Streamlit setup
st.set_page_config(page_title=lang["title"], layout="centered")
st.title(lang["title"])
st.subheader(lang["subtitle"])

uploaded_file = st.file_uploader(lang["upload"], type=["jpg", "jpeg", "png"])
MAX_FILE_SIZE_MB = 5

# Resize helper with letterboxing
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

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, left, top

if uploaded_file:
    uploaded_file.seek(0, os.SEEK_END)
    file_size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(lang["error"].format(file_size_mb, MAX_FILE_SIZE_MB))
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption=lang["image_caption"], use_container_width=True)

        if st.button(lang["button"]):
            with st.spinner(lang["spinner"]):
                resized_img, scale, pad_x, pad_y = resize_with_aspect(image)
                results = model.predict(resized_img, imgsz=640)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                small_count = 0
                big_count = 0
                AREA_THRESHOLD = 2500

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    area = (x2 - x1) * (y2 - y1)
                    label = lang["label_big"] if area >= AREA_THRESHOLD else lang["label_small"]
                    color = (0, 255, 0) if label == lang["label_big"] else (0, 255, 255)

                    if label == lang["label_big"]:
                        big_count += 1
                    else:
                        small_count += 1

                    cv2.rectangle(resized_img, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(resized_img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                total = big_count + small_count
                big_pct = (big_count / total * 100) if total else 0
                small_pct = (small_count / total * 100) if total else 0

                result_path = "resultado.jpg"
                cv2.imwrite(result_path, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

            # Display results
            st.markdown(f"<h2 style='text-align: center; color: green;'>{lang['result_title'].format(total)}</h2>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:lightgreen; font-size:18px;'>{lang['big'].format(big_count, big_pct)}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:orange; font-size:18px;'>{lang['small'].format(small_count, small_pct)}</span>", unsafe_allow_html=True)
            st.image(result_path, caption=lang["result_image"], use_container_width=True)
        ########################################################################################
            st.markdown("### 📝 Save this result to database")

with st.form("save_form"):
    id_area = st.text_input("🔹 Area ID (id_area)")
    id_centro = st.text_input("🔹 Center ID (id_centro)")
    id_linea = st.text_input("🔹 Line ID (id_linea)")
    id_temporada = st.text_input("🔹 Season ID (id_temporada)")
    n_muestra = st.number_input("🧪 Sample number", min_value=1, step=1)
    talla = st.number_input("📏 Average size (manual)", min_value=1)
    fecha_desperche = st.date_input("📅 Desperche Date (optional)")
    submit = st.form_submit_button("💾 Save to Database")

    if submit:
        conn = connect_to_db()
        if conn:
            cursor = conn.cursor()
            try:
                query = """
                    INSERT INTO reg_muestreo_siembra (id_area, id_centro, id_linea, id_temporada, n_muestra, cantidad, talla, fecha_desperche)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    id_area,
                    id_centro,
                    id_linea,
                    id_temporada,
                    n_muestra,
                    total,
                    talla,
                    fecha_desperche.strftime('%Y-%m-%d') if fecha_desperche else None
                ))
                conn.commit()
                st.success("✅ Record saved successfully!")
            except Error as e:
                st.error(f"❌ Error: {e}")
            finally:
                cursor.close()
                conn.close()
##################################################################################
if st.sidebar.button("📄 View Saved Records"):
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM reg_muestreo_siembra ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        st.table(rows)
        cursor.close()
        conn.close()

#################################################################################333

