import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

app = Flask(__name__)

# File Paths
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

        if len(subject_data) < 6:  # Not enough data for LSTM
            continue

        # Normalize attendance data
        min_attendance = subject_data['Attendance'].min()
        max_attendance = subject_data['Attendance'].max()
        
        if max_attendance == min_attendance:
            subject_data['Normalized'] = 1.0
        else:
            subject_data['Normalized'] = (
                (subject_data['Attendance'] - min_attendance) / 
                (max_attendance - min_attendance)
            )

        X, y = [], []
        window_size = 5  # Use last 5 days to predict next day

        for i in range(len(subject_data) - window_size):
            X.append(subject_data['Normalized'].iloc[i:i+window_size].values)
            y.append(subject_data['Normalized'].iloc[i+window_size])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM Model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, 1)),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, verbose=0)

        # Predict next 7 days
        last_window = X[-1]
        future_predictions = []
        
        for _ in range(7):
            next_day = model.predict(last_window.reshape(1, window_size, 1), verbose=0)[0][0]
            actual_pred = (next_day * (max_attendance - min_attendance)) + min_attendance
            future_predictions.append(actual_pred)
            last_window = np.roll(last_window, -1)
            last_window[-1] = next_day

        predictions[subject] = future_predictions

    return predictions

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    """Enter registration number"""
    if request.method == 'POST':
        return attendance_report()
    return render_template('attendance.html')

@app.route('/report', methods=['POST'])
def attendance_report():
    """Generate attendance report with LSTM-based predictions"""
    error_message = None
    graph_urls = []

    reg_no = request.form.get('reg_no', '').strip()
    if not reg_no:
        error_message = 'Please enter a valid Registration Number.'
    else:
        if not os.path.exists(EXCEL_FILE):
            error_message = f'File not found: {EXCEL_FILE}'
        else:
            try:
                df = pd.read_excel(EXCEL_FILE)
                df['Regn No.'] = df['Regn No.'].astype(str).str.strip()
                student_data = df[df['Regn No.'] == reg_no]

                if student_data.empty:
                    error_message = f'No records found for Registration No: {reg_no}'
                else:
                    # Generate Graphs
                    bar_url = generate_bar_chart(student_data, reg_no)
                    line_url = generate_line_chart(student_data, reg_no)
                    pie_url = generate_pie_chart(student_data, reg_no) if 'Status' in df.columns else None
                    
                    graph_urls.extend(filter(None, [bar_url, line_url, pie_url]))

                    # LSTM Predictions
                    predictions = prepare_lstm_data(df)
                    lstm_url = generate_lstm_chart(predictions, reg_no)
                    graph_urls.append(lstm_url)

            except Exception as e:
                error_message = f'Error processing file: {e}'

    return render_template('report.html', error=error_message, graph_urls=graph_urls)

def generate_bar_chart(student_data, reg_no):
    """Generate bar chart for subject-wise attendance"""
    subject_counts = student_data['Subject'].value_counts()

    plt.figure(figsize=(8, 6))
    plt.bar(subject_counts.index, subject_counts.values, color='skyblue', edgecolor='black')
    plt.xlabel('Subjects')
    plt.ylabel('Classes Attended')
    plt.title(f'Subject-wise Attendance for {reg_no}')
    plt.xticks(rotation=45, ha='right')

    graph_path = os.path.join(GRAPH_FOLDER, f'bar_{reg_no}.png')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return url_for('static', filename=f'graphs/bar_{reg_no}.png')

def generate_pie_chart(student_data, reg_no):
    """Generate pie chart for attendance distribution"""
    present_count = (student_data['Status'].str.lower() == 'present').sum()
    absent_count = (student_data['Status'].str.lower() == 'absent').sum()

    if present_count + absent_count == 0:
        return None  

    labels = ['Present', 'Absent']
    sizes = [present_count, absent_count]
    colors = ['#4CAF50', '#F44336']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title(f'Attendance Distribution for {reg_no}')

    graph_path = os.path.join(GRAPH_FOLDER, f'pie_{reg_no}.png')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return url_for('static', filename=f'graphs/pie_{reg_no}.png')

def generate_line_chart(student_data, reg_no):
    """Generate line chart for attendance trend"""
    student_data['Date'] = pd.to_datetime(student_data['Date'])
    student_data.sort_values('Date', inplace=True)

    attendance_counts = student_data.groupby('Date').size()

    plt.figure(figsize=(8, 6))
    plt.plot(attendance_counts.index, attendance_counts.values, marker='o', linestyle='-', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Classes Attended')
    plt.title(f'Attendance Trend for {reg_no}')
    plt.xticks(rotation=45)

    graph_path = os.path.join(GRAPH_FOLDER, f'line_{reg_no}.png')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return url_for('static', filename=f'graphs/line_{reg_no}.png')

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

    graph_path = os.path.join(GRAPH_FOLDER, f'lstm_{reg_no}.png')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return url_for('static', filename=f'graphs/lstm_{reg_no}.png')

if __name__ == '__main__':
    app.run(debug=True)
