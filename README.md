# Face Recognition Attendance System

This is a face recognition-based attendance system that allows you to:
- Add new students with their face data
- Check-in students using face recognition
- Check-out students using face recognition
- Store attendance records in a CSV file

## Requirements

- Python 3.7 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python attendance_system.py
```

2. The system provides four options:
   - Add Student: Register a new student with their face data
   - Check-In: Record student attendance using face recognition
   - Check-Out: Record student departure using face recognition
   - Exit: Close the application

3. When capturing face data:
   - Look directly at the camera
   - Press SPACE to capture the image
   - Press Q to cancel

## Data Storage

- Student data (face encodings, names, roll numbers) is stored in `student_data.json`
- Attendance records are stored in `attendance.csv`

## Notes

- Ensure good lighting conditions for accurate face recognition
- Keep your face centered in the camera frame
- The system requires a webcam to function 