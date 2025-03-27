1. Project Title
# Student Attendance Report Dashboard using LSTM

2. Project Description
This Flask-based web application allows students to check their attendance records and visualize their attendance trends. The system uses an LSTM (Long Short-Term Memory) model to predict future attendance trends based on historical data.

3. Features
✅ Student attendance record retrieval.
✅ Subject-wise attendance visualization (bar chart).
✅ LSTM-based future attendance prediction (line chart).
✅ Interactive Flask web interface.

4. Installation
Prerequisites
Ensure you have Python installed and required dependencies.

Clone the Repository
bash
Copy
Edit
git clone https://github.com/BuragapuHarika/STUDENT-BASED-ATTENDANCEN-REPOR-DASHBOARD-USING-DEEP-LEARNING-MODELS-LSTM-.git
cd STUDENT-BASED-ATTENDANCEN-REPOR-DASHBOARD-USING-DEEP-LEARNING-MODELS-LSTM-
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Run the Flask Application
bash
Copy
Edit
python app.py
5. Usage
Open your browser and go to http://127.0.0.1:5000/.

Enter a registration number to view attendance details.

View subject-wise attendance in a bar chart.

Get LSTM-based predictions for future attendance.

6. Folder Structure
csharp
Copy
Edit
├── static/
│   ├── graphs/        # Stores generated graphs
│   ├── styles.css     # CSS for styling
├── templates/
│   ├── index.html     # Home page
│   ├── attendance.html # Input page
│   ├── report.html    # Attendance report page
├── Attendance.xlsx    # Sample attendance dataset
├── app.py            # Flask application
├── requirements.txt   # Required dependencies
├── README.md          # Documentation
7. Technologies Used
Flask – Web framework

TensorFlow/Keras – LSTM model for prediction

Pandas – Data manipulation

Matplotlib – Data visualization

8. Contributors
👨‍💻 Developed by Harika

9. License
This project is open-source under the MIT License.
