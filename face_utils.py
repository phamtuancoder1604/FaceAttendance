import cv2 
import numpy as np
import os
import time
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sqlite3
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt

class Face: 
    def __init__(self, student_id=None):
        self.student_id = student_id 
        self.proto = 'model/deploy.prototxt'
        self.face_model =  'model/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        self.shape_model  = 'model/openface.nn4.small2.v1.t7'
        self.confidence = None
        
    def face_detection(self, img):
        model_face = cv2.dnn.readNetFromCaffe(self.proto, self.face_model)
        model_shape = cv2.dnn.readNetFromTorch(self.shape_model)
        h, w = img.shape[:2]
        blob_img = cv2.dnn.blobFromImage(img, 1.0, (300,300), (123,175,106), swapRB=False)
        model_face.setInput(blob_img)
        faces = model_face.forward()
        i = np.argmax(faces[0,0,:,2])
        self.confidence = faces[0,0,i,2]
        if self.confidence > 0.9:
            box = faces[0,0,i,3:7] * np.array([w,h,w,h])
            startx, starty, endx, endy = box.astype('int')
            roi = img[starty:endy, startx:endx]
            roi_blob = cv2.dnn.blobFromImage(img, 1/255, (96,96), (0,0,0), swapRB=False, crop=False)
            model_shape.setInput(roi_blob)
            model_shape.forward()
        return img
    
    def face_shape_detection(self, img_path):
        model_face = cv2.dnn.readNetFromCaffe(self.proto, self.face_model)
        model_shape = cv2.dnn.readNetFromTorch(self.shape_model)
        img1 = cv2.imread(img_path)
        img = img1.copy()
        h, w = img.shape[:2]
        blob_img = cv2.dnn.blobFromImage(img, 1.0, (300,300), (123,175,106), swapRB=False)
        model_face.setInput(blob_img)
        faces = model_face.forward()
        i = np.argmax(faces[0,0,:,2])
        self.confidence = faces[0,0,i,2]
        if self.confidence > 0.9:
            box = faces[0,0,i,3:7] * np.array([w,h,w,h])
            startx, starty, endx, endy = box.astype('int')
            roi = img[starty:endy, startx:endx]
            roi_blob = cv2.dnn.blobFromImage(img, 1/255, (96,96), (0,0,0), swapRB=False, crop=False)
            model_shape.setInput(roi_blob)
            vectors = model_shape.forward()
            return vectors
        return None
    
    def get_face(self):
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Windows', cv2.WINDOW_NORMAL)
        count = 0 
        folder_path = f'student_images/{self.student_id}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            print(f'Folder {self.student_id} is exist')
        while True: 
            ret, frame = camera.read()
            if not ret:
                break 
            count += 1 
            frame = self.face_detection(frame)
            if count % 2 == 0 and self.confidence > 0.95:
                img_path = folder_path + '/' + f'img_{count}.jpg'
                cv2.imwrite(img_path, frame)
            text = f'Face {self.confidence * 100:.2f}%'
            cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            cv2.imshow('Windows', frame)
            if cv2.waitKey(10) == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()
    
    def data(self):
        vector_data = dict(data=[], label=[])
        mydir = 'student_images'
        folders = os.listdir(mydir)
        for folder in folders:
            path = mydir + '/' + folder 
            files = os.listdir(path)
            for file in files:
                try:
                    vector = self.face_shape_detection(path + '/' + file)
                    if vector is not None:
                        vector_data['data'].append(vector)
                        vector_data['label'].append(folder)  # folder = student_id
                except:
                    pass
        pickle.dump(vector_data, open('Results.pickle', mode='wb'))
    
    def model(self):
        data = pickle.load(open('Results.pickle', 'rb'))
        x = np.array(data['data']).reshape(len(data['data']), -1)
        y = np.array(data['label'])
        le = LabelEncoder()
        y = le.fit_transform(y)
        pickle.dump(le, open('Encode.pkl', mode='wb'))
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)
        model_voting = VotingClassifier(estimators=[
            ('logistic', LogisticRegression(max_iter=500)),
            ('svm', SVC(probability=True)),
            ('rf', RandomForestClassifier(n_estimators=20))
        ], voting='soft', weights=[2,3,1])
        param_grid = {
            'svm__C': [1, 3, 5],
            'svm__gamma': [0.1, 0.3, 0.5],
            'rf__n_estimators': [10, 20, 30],
            'rf__max_depth': [3, 5, 7]
        }
        model_grid = GridSearchCV(model_voting, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)
        model_grid.fit(x_train, y_train)
        model = model_grid.best_estimator_
        pickle.dump(model, open("ml_face_person_identity.pkl", mode='wb'))
    
    def predict_face(self, img):
        model_face = cv2.dnn.readNetFromCaffe(self.proto, self.face_model)
        model_shape = cv2.dnn.readNetFromTorch(self.shape_model)
        ml_model = pickle.load(open("ml_face_person_identity.pkl", mode='rb'))
        le = pickle.load(open("Encode.pkl", mode='rb'))
        machinelearning_results = dict(face_detect_score=[], face_name=[], face_name_score=[], count=[])
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), (123,175,104), swapRB=False, crop=False)
        model_face.setInput(blob)
        faces = model_face.forward()
        if len(faces) > 0:
            i = faces[0,0,:,2].argmax()
            confidence = faces[0,0,i,2]
            count = 0 
            if confidence > 0.95:
                box = faces[0,0,i,3:7] * np.array([w,h,w,h])
                start_x, start_y, end_x, end_y = box.astype('int')
                cv2.rectangle(img, (start_x,start_y), (end_x,end_y), (0,255,0), 2) 
                roi = img[start_y:end_y, start_x:end_x]
                roi_blob = cv2.dnn.blobFromImage(roi, 1/255, (96,96), (255,255,255), swapRB=True, crop=True)
                model_shape.setInput(roi_blob)
                vectors = model_shape.forward()
                face_name_label = ml_model.predict(vectors)[0]
                student_id_pred = le.inverse_transform([face_name_label])[0]
                face_name_score = ml_model.predict_proba(vectors).max()
                machinelearning_results['count'] = count 
                machinelearning_results['face_name'] = student_id_pred
                machinelearning_results['face_detect_score'] = confidence 
                machinelearning_results['face_name_score'] = face_name_score 
                count += 1
        return img, machinelearning_results
    
    def display(self):
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Windows', cv2.WINDOW_NORMAL)
        count = 0 
        while True: 
            ret, frame = camera.read()
            if not ret:
                break 
            frame, ml = self.predict_face(frame)
            if 'face_name_score' in ml and isinstance(ml['face_name_score'], (int, float, np.floating)):
                face_name_score = ml['face_name_score'] * 100
                text = f'Face {face_name_score:.2f}%'
            else:
                face_name_score = 0  
                text = "No face detected"
            student_id_pred = ml.get('face_name', 'Unknown')  
            text_name = f"Student ID: {student_id_pred}"
            text_score = f"Confidence: {face_name_score:.2f}%"
            cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            cv2.putText(frame, text_name, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            if face_name_score > 87: 
                cv2.putText(frame, 'Complete Valid', (70,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3)
            cv2.imshow('Windows', frame)
            if face_name_score > 87: 
                cv2.waitKey(2000)
                break 
            if cv2.waitKey(10) == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()
