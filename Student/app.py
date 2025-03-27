import os

import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend if needed
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# ✅ Use the Absolute Path to your Attendance Excel File
EXCEL_FILE = r"D:\Student (2)\Student\Attendance.xlsx"

# Folder for saving graphs
GRAPH_FOLDER = "static/graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True)

@app.route("/")
def home():
    """
    Renders the home page, which has a card and a 'View Report' button.
    """
    return render_template("index.html")

@app.route("/attendance", methods=["GET", "POST"])
def attendance_page():
    """
    Renders the attendance page and processes attendance data when the user submits the form.
    """
    if request.method == "POST":
        # Redirect the POST request data to the /report route
        return attendance_report()
    return render_template("attendance.html")

@app.route("/report", methods=["GET", "POST"])
def attendance_report():
    """
    Fetches and analyzes attendance data from Excel, generates graphs, and displays the report.
    """
    error_message = None
    graph_urls = []
    analysis = []

    if request.method == "POST":
        # Get the registration number from the form
        search_query = request.form.get("reg_no", "").strip()

        if not search_query:
            error_message = "Please enter a valid Registration Number."
        else:
            try:
                # ✅ Check if the Excel file exists before reading
                if not os.path.exists(EXCEL_FILE):
                    raise FileNotFoundError(f"File not found: {EXCEL_FILE}")

                # Load Excel file
                df = pd.read_excel(EXCEL_FILE)

                # Convert 'Regn No.' to string & remove trailing ".0"
                df["Regn No."] = df["Regn No."].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

                # Filter rows for the specified registration number
                student_data = df[df["Regn No."] == search_query]

                if student_data.empty:
                    error_message = f"No records found for Registration No: {search_query}"
                else:
                    # Subject-wise attendance
                    subject_counts = student_data["Subject"].value_counts().reset_index()
                    subject_counts.columns = ["Subject", "Attended"]

                    # Overall total classes per subject
                    total_classes = df["Subject"].value_counts().reset_index()
                    total_classes.columns = ["Subject", "Total"]

                    # Merge them to calculate attendance percentage
                    analysis_df = pd.merge(subject_counts, total_classes, on="Subject", how="right").fillna(0)
                    analysis_df["Attendance Percentage"] = (
                        analysis_df["Attended"] / analysis_df["Total"] * 100
                    ).round(2)

                    # Convert to a list of dicts for easy template rendering
                    analysis = analysis_df.to_dict(orient="records")

                    # Generate Graphs
                    bar_chart_url = generate_bar_chart(student_data, search_query)
                    if bar_chart_url:
                        graph_urls.append(bar_chart_url)

                    pie_chart_url = generate_pie_chart(student_data, search_query)
                    if pie_chart_url:
                        graph_urls.append(pie_chart_url)

                    line_chart_url = generate_line_chart(student_data, search_query)
                    if line_chart_url:
                        graph_urls.append(line_chart_url)

            except Exception as e:
                error_message = f"Error processing file: {e}"

    # Render the report template with the results
    return render_template(
        "report.html",
        error=error_message,
        analysis=analysis,
        graph_urls=graph_urls
    )

def generate_bar_chart(student_data, reg_no):
    """Generates a bar chart showing how many classes were attended for each subject."""
    if "Subject" not in student_data.columns:
        return None

    subject_counts = student_data["Subject"].value_counts()
    if subject_counts.empty:
        return None

    plt.figure(figsize=(6, 4))
    subject_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.xlabel("Subjects")
    plt.ylabel("Classes Attended")
    plt.title(f"Subject-wise Attendance for {reg_no}")
    plt.xticks(rotation=45, ha='right')

    graph_path = os.path.join(GRAPH_FOLDER, f"bar_{reg_no}.png")
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return url_for('static', filename=f'graphs/bar_{reg_no}.png')

def generate_pie_chart(student_data, reg_no):
    """Generates a pie chart showing the distribution of attended classes across subjects."""
    if "Subject" not in student_data.columns:
        return None

    subject_counts = student_data["Subject"].value_counts()
    if subject_counts.empty:
        return None

    plt.figure(figsize=(4, 4))
    subject_counts.plot(kind="pie", autopct='%1.1f%%')
    plt.title(f"Attendance Distribution for {reg_no}")
    plt.ylabel("")  # Hide default y-label for pie chart

    graph_path = os.path.join(GRAPH_FOLDER, f"pie_{reg_no}.png")
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return url_for('static', filename=f'graphs/pie_{reg_no}.png')

def generate_line_chart(student_data, reg_no):
    """
    Generates a line chart to show attendance over time (grouped by date).
    This requires a 'Date' column in the Excel file.
    """
    if "Date" not in student_data.columns:
        return None

    # Convert Date column to datetime if not already
    student_data["Date"] = pd.to_datetime(student_data["Date"], errors='coerce')
    # Drop rows that don't have valid dates
    student_data = student_data.dropna(subset=["Date"])

    if student_data.empty:
        return None

    # Group by Date and count attendance
    attendance_trend = student_data.groupby("Date").size().sort_index()

    plt.figure(figsize=(6, 4))
    plt.plot(attendance_trend.index, attendance_trend.values, marker="o", linestyle="-", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Attendance Count")
    plt.title(f"Attendance Trend for {reg_no}")
    plt.xticks(rotation=45, ha='right')

    graph_path = os.path.join(GRAPH_FOLDER, f"line_{reg_no}.png")
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return url_for('static', filename=f'graphs/line_{reg_no}.png')

if __name__ == "__main__":
    app.run(debug=True)
