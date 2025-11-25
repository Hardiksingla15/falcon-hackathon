import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Space Station Safety Detector",
    layout="wide",
    page_icon="🚀"
)

# Load Model
model = YOLO("../best.pt")

# -----------------------------------------------------------
# SIDEBAR — SETTINGS + ANALYTICS
# -----------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        0.10, 0.90, 0.25, 0.01,
        help="Lower = detect more objects, Higher = stricter"
    )

    st.markdown("---")
    st.markdown("## 📊 Model Evaluation Summary")

    map_50 = 0.801
    map_50_95 = 0.659
    combined_score = round((map_50 + map_50_95) / 2, 3)

    st.write(f"**mAP@0.5:** `{map_50}`")
    st.write(f"**mAP@0.5:0.95:** `{map_50_95}`")
    st.write(f"**Overall Model Score:** `{combined_score}`")

    st.markdown("---")
    det_header = st.empty()
    sidebar_objects = st.empty()
    sidebar_conf_summary = st.empty()

# -----------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------
st.markdown("""
<h1 style='text-align:center; color:#FF4B4B;'>
🚀 Space Station Safety Object Detector
</h1>
<p style='text-align:center; font-size:18px; color:#CCCCCC;'>
Detect OxygenTank, NitrogenTank, FirstAidBox, FireAlarm, SafetySwitchPanel, EmergencyPhone, FireExtinguisher
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1.5])

# -------------------
# Upload Section
# -------------------
with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Upload Space Station Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        input_img = Image.open(uploaded_file)
        st.image(input_img, caption="Uploaded Image", use_container_width=True)
    else:
        input_img = None

# -------------------
# Detection Section
# -------------------
with col2:
    st.subheader("📦 Detection Result")
    output_area = st.empty()

run_btn = st.button("Run Detection", use_container_width=True)

# -----------------------------------------------------------
# RUN INFERENCE
# -----------------------------------------------------------
if run_btn:
    if input_img is None:
        st.warning("Please upload an image first.")
    else:
        # Progress bar animation
        progress = st.progress(0)
        for i in range(80):
            time.sleep(0.01)
            progress.progress(i + 1)

        results = model.predict(
            source=np.array(input_img),
            conf=conf_threshold
        )

        # SHOW IMAGE WITH BOXES
        plotted_img = results[0].plot()
        output_area.image(plotted_img, caption="Detected Objects", use_container_width=True)

        # -------------------------
        # SIDEBAR — OBJECT ANALYTICS
        # -------------------------
        detections = results[0].boxes
        names = results[0].names

        det_header.markdown("## 🟢 Objects Detected:")

        if len(detections) == 0:
            sidebar_objects.write("No objects detected.")
            sidebar_conf_summary.write("")
        else:
            object_text = ""
            conf_values = []

            for box in detections:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = names[cls]

                conf_values.append(conf)
                object_text += f"**{label} — `{round(conf, 3)}` confidence**\n\n"

            sidebar_objects.markdown(object_text)

            # Confidence Summary
            avg_conf = round(sum(conf_values) / len(conf_values), 3)
            max_conf = round(max(conf_values), 3)

            summary_md = f"""
            ### 🧪 Image Confidence Summary  
            - **Average Confidence:** `{avg_conf}`  
            - **Highest Confidence:** `{max_conf}`  
            """

            sidebar_conf_summary.markdown(summary_md)
