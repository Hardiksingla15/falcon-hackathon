🚀 Space Station Safety Object Detector

AI system for detecting OxygenTank, NitrogenTank, FirstAidBox, FireAlarm, SafetySwitchPanel, EmergencyPhone, FireExtinguisher
Built using DualityAI Falcon Synthetic Dataset + YOLOv8 + Custom Training.

⭐ Features

Real-time image detection

Clean, responsive UI using Streamlit

Confidence score per object

Summary panel (Average Confidence + Highest Confidence)

mAP metrics included

Uses your fine-tuned model (best.pt)

📁 Project Structure
falcon-hackathon/
│── best.pt
│── ui/app.py
│── requirements.txt
│── README.md

🖼️ Demo Screenshot

(Add the UI screenshot you showed me)

🔧 How to Run
pip install -r requirements.txt
cd ui
streamlit run app.py

🧠 Model Details

Architecture: YOLOv8n

Epochs: 45 (40 + fine-tuning 5)

mAP@0.5: 0.80

mAP@0.5:0.95: 0.65

📊 UI Features

Drag & drop image upload

Detection result preview

Object confidence scores

Combined model score

Adjustable confidence threshold

👥 Team

Hardik & Team – Build with India Hackathon 2025
