import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

app = Flask(__name__)

# Paths
EXCEL_FILE = os.path.join(os.path.dirname(__file__), 'Attendance.xlsx')
GRAPH_FOLDER = os.path.join('static', 'graphs')
os.makedirs(GRAPH_FOLDER, exist_ok=True)

def prepare_lstm_data(df):
    """Prepare data for LSTM training and prediction"""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)

    subjects = df['Subject'].unique()
    predictions = {}

    for subject in subjects:
        subject_data = df[df['Subject'] == subject]
        subject_data = subject_data.groupby('Date').size().reset_index(name='Attendance')

        if len(subject_data) < 6:
            continue

        # Normalize data
        min_att, max_att = subject_data['Attendance'].min(), subject_data['Attendance'].max()
        subject_data['Normalized'] = (subject_data['Attendance'] - min_att) / (max_att - min_att) if max_att != min_att else 1.0

        X, y, window_size = [], [], 5
        for i in range(len(subject_data) - window_size):
            X.append(subject_data['Normalized'].iloc[i:i+window_size].values)
            y.append(subject_data['Normalized'].iloc[i+window_size])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, 1)),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, verbose=0)

        # Predict next 7 days
        last_window, future_predictions = X[-1], []
        for _ in range(7):
            next_day = model.predict(last_window.reshape(1, window_size, 1), verbose=0)[0][0]
            future_predictions.append((next_day * (max_att - min_att)) + min_att)
            last_window = np.roll(last_window, -1)
            last_window[-1] = next_day

        predictions[subject] = future_predictions
    
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        return attendance_report()
    return render_template('attendance.html')

@app.route('/report', methods=['POST'])
def attendance_report():
    error_message, attendance_records, graph_urls = None, [], []
    reg_no = request.form.get('reg_no', '').strip()
    
    if not reg_no:
        error_message = 'Please enter a valid Registration Number.'
    elif not os.path.exists(EXCEL_FILE):
        error_message = f'File not found: {EXCEL_FILE}'
    else:
        try:
            df = pd.read_excel(EXCEL_FILE)
            df['Regn No.'] = df['Regn No.'].astype(str).str.strip()
            student_data = df[df['Regn No.'] == reg_no]

            if student_data.empty:
                error_message = f'No records found for Registration No: {reg_no}'
            else:
                attendance_records = student_data.to_dict(orient='records')
                predictions = prepare_lstm_data(df)
                
                graph_urls = [
                    generate_bar_chart(student_data, reg_no),
                    generate_pie_chart(student_data, reg_no),
                    generate_lstm_chart(predictions, reg_no)
                ]
        except Exception as e:
            error_message = f'Error processing file: {e}'
    
    return render_template('report.html', error=error_message, attendance_records=attendance_records, graph_urls=graph_urls)

def generate_bar_chart(student_data, reg_no):
    """Generate bar chart for subject-wise attendance"""
    subject_counts = student_data['Subject'].value_counts()
    
    plt.figure(figsize=(8, 6))
    plt.bar(subject_counts.index, subject_counts.values, color='skyblue', edgecolor='black')
    plt.xlabel('Subjects')
    plt.ylabel('Classes Attended')
    plt.title(f'Subject-wise Attendance for {reg_no}')
    plt.xticks(rotation=45, ha='right')
    
    return save_graph(f'bar_{reg_no}.png')

def generate_pie_chart(student_data, reg_no):
    """Generate pie chart for subject-wise attendance"""
    subject_counts = student_data['Subject'].value_counts()
    
    plt.figure(figsize=(8, 6))
    plt.pie(subject_counts.values, labels=subject_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.title(f'Subject-wise Attendance Distribution for {reg_no}')
    
    return save_graph(f'pie_{reg_no}.png')

def generate_lstm_chart(predictions, reg_no):
    """Generate line chart for LSTM-based predictions"""
    plt.figure(figsize=(8, 6))
    
    for subject, future_attendance in predictions.items():
        plt.plot(range(1, 8), future_attendance, marker='o', linestyle='-', label=subject)
    
    plt.xlabel('Future Days')
    plt.ylabel('Predicted Attendance')
    plt.title(f'Predicted Attendance Trends for {reg_no}')
    plt.xticks(range(1, 8))
    plt.legend()
    
    return save_graph(f'lstm_{reg_no}.png')

def save_graph(filename):
    """Save and return graph URL"""
    graph_path = os.path.join(GRAPH_FOLDER, filename)
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()
    return url_for('static', filename=f'graphs/{filename}')

if __name__ == '__main__':
    app.run(debug=True)
