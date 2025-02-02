# Music Recommendation System

## Description

This project is a Flask-based application designed to provide personalized recommendations by utilizing grey wolf optimizer and implicit library. It leverages machine learning algorithms to analyze user data and generate tailored suggestions.

## Setup Instructions

### Prerequisites

- Python 3.x installed on your machine
- `pip` package manager

### Setting Up the Virtual Environment

1. **Create a Virtual Environment:**

   Open your terminal and navigate to the project directory. Run the following command to create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**

   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install Required Packages:**

   With the virtual environment activated, install the necessary packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Flask Application

1. **Set the Flask App Environment Variable:**

   Set the `FLASK_APP` environment variable to point to your main application file. For example, if your main file is `app.py`, run:

   ```bash
   export FLASK_APP=app.py
   ```

   On Windows, use:

   ```bash
   set FLASK_APP=app.py
   ```

2. **Run the Flask Application:**

   Start the Flask development server by running:

   ```bash
   flask run
   ```

   The application will be accessible at `http://127.0.0.1:5000/`.
