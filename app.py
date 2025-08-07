import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
import mysql.connector
import pandas as pd
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
# Ensure the model path is correct
try:
    model = YOLO('yolov8_model/best.pt')
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()


# ------------------------- DB Helper Functions -------------------------
def get_db_connection():
    """Establishes and returns a database connection."""
    return mysql.connector.connect(**db_config)

def ensure_schema():
    """
    Checks if the target table has the necessary columns for detailed measurements.
    If not, it alters the table to add them. This makes the app robust to DB changes.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW COLUMNS FROM reg_muestreo_siembra")
        existing_columns = [col[0] for col in cursor.fetchall()]

        alter_commands = []
        # CORRECTED: Added check for the timestamp column
        if 'fecha_registro' not in existing_columns:
            alter_commands.append("ADD COLUMN fecha_registro DATETIME")
        if 'length_mm' not in existing_columns:
            alter_commands.append("ADD COLUMN length_mm FLOAT")
        if 'height_mm' not in existing_columns:
            alter_commands.append("ADD COLUMN height_mm FLOAT")
        if 'peso_gr' not in existing_columns:
            alter_commands.append("ADD COLUMN peso_gr FLOAT")
        if 'units' not in existing_columns:
            # This column will store a description of units used, e.g., "mm/g"
            alter_commands.append("ADD COLUMN units VARCHAR(50)")

        if alter_commands:
            st.info("Updating database schema...")
            alter_stmt = f"ALTER TABLE reg_muestreo_siembra {', '.join(alter_commands)}"
            cursor.execute(alter_stmt)
            conn.commit()
            st.success("Database schema updated successfully!")

        conn.close()
    except Exception as e:
        st.error("❌ Database schema check failed.")
        st.exception(e)

# --- Run schema check at app startup ---
ensure_schema()


# ------------------------- Language selection -------------------------
language = st.sidebar.selectbox("🌐 Language / Idioma", ["Español", "English"])
lang = {
    "English": {
        "title": "🐚 Mussel Detector using YOLOv8",
        "subtitle": "Upload an image of mussels (JPG/PNG)",
        "upload": "Choose an image",
        "button": "🔍 Count Mussels in the Image",
        "save_button": "✅ Save to Database",
        "save_subheader": "💾 Save Results",
        "result_title": "✅ {} mussels detected",
        "big": "🟩 Big mussels: {} ({:.1f}%)",
        "small": "🟨 Small mussels: {} ({:.1f}%)",
        "manual_entry_header": "✍️ Manual Entry (Optional)",
        "length_label": "📏 Length",
        "height_label": "📐 Height",
        "weight_label": "⚖️ Weight",
        "unit_length_label": "📏 Length Unit",
        "unit_weight_label": "⚖️ Weight Unit",
        "manual_count_label": "🔢 Mussel Count (Manual Override)",
        "save_success": "✅ Data saved to database!",
        "save_error": "❌ Failed to save entry.",
        "db_preview_header": "🔍 Latest 10 entries from `reg_muestreo_siembra`",
        "db_preview_checkbox": "🧪 Show Table Preview",
        "db_load_error": "❌ Failed to load table preview",
        "label_big": "big",
        "label_small": "small",
    },
    "Español": {
        "title": "🐚 Contador de Semillas Standrews",
        "subtitle": "Sube una imagen de mejillones (JPG/PNG)",
        "upload": "Elige una imagen",
        "button": "🔍 Contar Mejillones en la Imagen",
        "save_button": "✅ Guardar en la Base de Datos",
        "save_subheader": "💾 Guardar Resultados",
        "result_title": "✅ {} mejillones detectados",
        "big": "🟩 Mejillones grandes: {} ({:.1f}%)",
        "small": "🟨 Mejillones pequeños: {} ({:.1f}%)",
        "manual_entry_header": "✍️ Ingreso Manual (Opcional)",
        "length_label": "📏 Largo",
        "height_label": "📐 Alto",
        "weight_label": "⚖️ Peso",
        "unit_length_label": "📏 Unidad de Largo",
        "unit_weight_label": "⚖️ Unidad de Peso",
        "manual_count_label": "🔢 Conteo de Mejillones (Manual)",
        "save_success": "✅ ¡Datos guardados en la base de datos!",
        "save_error": "❌ Falló al guardar el registro.",
        "db_preview_header": "🔍 Últimos 10 registros de `reg_muestreo_siembra`",
        "db_preview_checkbox": "🧪 Mostrar Vista Previa de la Tabla",
        "db_load_error": "❌ Falló al cargar la vista previa de la tabla",
        "label_big": "grande",
        "label_small": "pequeño",
    }
}[language]

st.set_page_config(page_title=lang["title"], layout="wide")
st.title(lang["title"])
st.subheader(lang["subtitle"])

# ------------------------- Data Fetching and Utility Functions -------------------------
@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_dropdown_data():
    """Fetches data for the dropdowns from the database."""
    conn = get_db_connection()
    areas = pd.read_sql("SELECT id_area, nombre_area FROM areas", conn)
    centros = pd.read_sql("SELECT id_centro, nombre_centro, area_id FROM centro_cultivo", conn)
    lineas = pd.read_sql("SELECT id_lineas, cod_lineas, centros_id FROM lineas", conn)
    temporadas = pd.read_sql("SELECT id_temporda, nombre FROM temporada", conn)
    conn.close()
    return areas, centros, lineas, temporadas

def resize_with_aspect(image, target_size=(1200, 1600)):
    """Resizes and pads an image to a target size while maintaining aspect ratio."""
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
    return padded

# ------------------------- Main App Logic -------------------------
uploaded_file = st.file_uploader(lang["upload"], type=["jpg", "jpeg", "png"])
MAX_FILE_SIZE_MB = 10

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"❌ File too large. Max allowed: {MAX_FILE_SIZE_MB} MB.")
    else:
        st.session_state["image_bytes"] = uploaded_file.read()
        image = Image.open(BytesIO(st.session_state["image_bytes"]))
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button(lang["button"]):
            with st.spinner('Processing...'):
                resized_img = resize_with_aspect(image)
                results = model.predict(resized_img, imgsz=640, conf=0.5) # Set confidence threshold
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                
                big_count = 0
                small_count = 0
                AREA_THRESHOLD = 2500 # Adjust as needed

                annotated_img = resized_img.copy()
                for (x1, y1, x2, y2) in boxes:
                    area = (x2 - x1) * (y2 - y1)
                    if area >= AREA_THRESHOLD:
                        big_count += 1
                        color = (0, 255, 0) # Green for big
                    else:
                        small_count += 1
                        color = (0, 255, 255) # Yellow for small
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                total = big_count + small_count
                result_path = "resultado.jpg"
                cv2.imwrite(result_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                
                # Save results to session state to persist across reruns
                st.session_state["result_image_path"] = result_path
                st.session_state["total"] = total
                st.session_state["big_count"] = big_count
                st.session_state["small_count"] = small_count

# ------------------------- Show Results and Save Form -------------------------
if "result_image_path" in st.session_state:
    st.image(st.session_state["result_image_path"], caption="Detection Result", use_container_width=True)
    total_mussels = st.session_state.get("total", 0)
    
    if total_mussels > 0:
        big_count = st.session_state.get("big_count", 0)
        small_count = st.session_state.get("small_count", 0)
        st.success(lang["result_title"].format(total_mussels))
        st.write(lang["big"].format(big_count, (big_count / total_mussels * 100)))
        st.write(lang["small"].format(small_count, (small_count / total_mussels * 100)))
    else:
        st.info("No mussels were detected in the image.")

    st.markdown("---")
    st.subheader(lang["save_subheader"])
    
    # --- Dynamic Dropdowns ---
    areas_df, centros_df, lineas_df, temporadas_df = get_dropdown_data()
    
    area_map = pd.Series(areas_df.nombre_area.values, index=areas_df.id_area).to_dict()
    selected_area_id = st.selectbox("🗺️ Area", options=list(area_map.keys()), format_func=lambda x: area_map[x])
    
    filtered_centros = centros_df[centros_df['area_id'] == selected_area_id]
    centro_map = pd.Series(filtered_centros.nombre_centro.values, index=filtered_centros.id_centro).to_dict()
    selected_centro_id = st.selectbox("🏗️ Centro de Cultivo", options=list(centro_map.keys()), format_func=lambda x: centro_map.get(x, "N/A"))

    filtered_lineas = lineas_df[lineas_df['centros_id'] == selected_centro_id]
    linea_map = pd.Series(filtered_lineas.cod_lineas.values, index=filtered_lineas.id_lineas).to_dict()
    selected_linea_id = st.selectbox("🧵 Línea", options=list(linea_map.keys()), format_func=lambda x: linea_map.get(x, "N/A"))

    temporada_map = pd.Series(temporadas_df.nombre.values, index=temporadas_df.id_temporda).to_dict()
    selected_temporada_id = st.selectbox("📅 Temporada", options=list(temporada_map.keys()), format_func=lambda x: temporada_map.get(x, "N/A"))
    
    # --- Manual Data Entry Form ---
    st.markdown(f"### {lang['manual_entry_header']}")
    
    col1, col2 = st.columns(2)
    with col1:
        length = st.number_input(lang["length_label"], min_value=0.0, step=0.1, format="%.1f")
        height = st.number_input(lang["height_label"], min_value=0.0, step=0.1, format="%.1f")
        weight = st.number_input(lang["weight_label"], min_value=0.0, step=0.1, format="%.1f")
    with col2:
        unit_length = st.selectbox(lang["unit_length_label"], ["mm", "cm", "in"])
        unit_weight = st.selectbox(lang["unit_weight_label"], ["g", "mg", "kg"])
        manual_count = st.number_input(lang["manual_count_label"], min_value=0, value=st.session_state.get("total", 0), step=1)
        
    if st.button(lang["save_button"]):
        # Use manual count if provided, otherwise use the detected count
        final_count = manual_count if manual_count > 0 else total_mussels
        # Combine units into a single descriptive string
        units_str = f"L/H: {unit_length}, W: {unit_weight}"

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # ** NEW INSERT STATEMENT FOR SEPARATE COLUMNS **
            sql = """
                INSERT INTO reg_muestreo_siembra (
                    fecha_registro, id_area, id_centro, id_linea, id_temporada,
                    cantidad, length_mm, height_mm, peso_gr, units
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                datetime.now(),
                selected_area_id,
                selected_centro_id,
                selected_linea_id,
                selected_temporada_id,
                final_count,
                length,
                height,
                weight,
                units_str
            )
            
            cursor.execute(sql, values)
            conn.commit()
            conn.close()
            st.success(lang["save_success"])
            
        except Exception as e:
            st.error(lang["save_error"])
            st.exception(e)

# ------------------------- Database Preview in Sidebar -------------------------
if st.sidebar.checkbox(lang["db_preview_checkbox"]):
    with st.expander(lang["db_preview_header"]):
        try:
            conn = get_db_connection()
            # Use pandas for a cleaner display
            df = pd.read_sql("SELECT id, fecha_registro, id_area, cantidad, length_mm, height_mm, peso_gr, units FROM reg_muestreo_siembra ORDER BY id DESC LIMIT 10", conn)
            st.dataframe(df)
            conn.close()
        except Exception as e:
            st.error(lang["db_load_error"])
            st.exception(e)


