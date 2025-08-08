import os
import cv2
import sqlite3
import pickle
import numpy as np
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

DB_FILE = "attendance.db"
IMG_DIR = "student_images"
MODEL_FILE = "ml_face_model.pkl"
ENCODER_FILE = "label_encoder.pkl"

#DATABASE 
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE,
                    name TEXT,
                    class_name TEXT,
                    image_dir TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    date TEXT,
                    status TEXT)""")
    conn.commit()
    conn.close()

#CRUD STUDENTS 
def add_student(student_id, name, class_name):
    image_dir = os.path.join(IMG_DIR, student_id)
    os.makedirs(image_dir, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("INSERT INTO students(student_id, name, class_name, image_dir) VALUES (?, ?, ?, ?)",
                     (student_id, name, class_name, image_dir))
        conn.commit()
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Student ID already exists")
    conn.close()

def delete_student(student_id):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM students WHERE student_id=?", (student_id,))
    conn.commit()
    conn.close()

def update_student(student_id, name, class_name):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE students SET name=?, class_name=? WHERE student_id=?", (name, class_name, student_id))
    conn.commit()
    conn.close()

def get_students():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT student_id, name, class_name FROM students").fetchall()
    conn.close()
    return rows

# ATTENDANCE 
def mark_attendance(student_id, status="Present"):
    date = datetime.date.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO attendance(student_id, date, status) VALUES (?, ?, ?)", (student_id, date, status))
    conn.commit()
    conn.close()

def show_report():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT date, COUNT(*) FROM attendance GROUP BY date").fetchall()
    conn.close()
    if not rows:   # THÊM DÒNG NÀY ĐỂ KIỂM TRA
        messagebox.showinfo("Thông báo", "Chưa có dữ liệu điểm danh để báo cáo.")
        return
    dates, counts = zip(*rows)
    plt.bar(dates, counts)
    plt.title("Attendance per day")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_attendance_history():
    import sqlite3
    conn = sqlite3.connect("attendance.db")
    rows = conn.execute("""
        SELECT attendance.date, students.student_id, students.name, students.class_name, attendance.status
        FROM attendance
        JOIN students ON attendance.student_id = students.student_id
        ORDER BY attendance.date DESC
    """).fetchall()
    conn.close()
    return rows




