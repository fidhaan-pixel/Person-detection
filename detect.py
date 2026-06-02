import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import sqlite3
from datetime import datetime
import smtplib

# Load model
model = YOLO("yolov8n.pt")

# Database
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users(
    username TEXT, password TEXT
)""")

# ---------------- LOGIN ----------------
st.title("🔐 Login System")

menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Signup":
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type='password')
    if st.button("Signup"):
        c.execute("INSERT INTO users VALUES (?,?)", (new_user, new_pass))
        conn.commit()
        st.success("Account created!")

elif choice == "Login":
    user = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (user, password))
        data = c.fetchone()

        if data:
            st.success("Login Successful ✅")

            # ---------------- DETECTION ----------------
            run = st.checkbox("Start Camera")

            cap = cv2.VideoCapture(0)
            unique_ids = set()
            history = []

            frame_window = st.image([])

            while run:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(frame, persist=True)

                if results[0].boxes.id is not None:
                    for box, track_id, cls in zip(
                        results[0].boxes.xyxy,
                        results[0].boxes.id,
                        results[0].boxes.cls
                    ):
                        if int(cls) == 0:
                            x1, y1, x2, y2 = map(int, box)
                            unique_ids.add(int(track_id))

                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                count = len(unique_ids)

                # Save history
                history.append([datetime.now().strftime("%H:%M:%S"), count])

                # ALERT (if crowd)
                if count > 3:
                    st.warning("⚠️ Crowd detected!")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame)

            cap.release()

            # Show data
            if history:
                df = pd.DataFrame(history, columns=["Time", "Count"])
                st.line_chart(df.set_index("Time"))

        else:
            st.error("Invalid login ❌")