import cv2
import numpy as np
import sqlite3
import threading
from queue import Queue
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import datetime
import os

class FaceRecognitionAttendance:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.known_rolls = []
        self.attendance_data = []
        self.is_running = False
        self.last_recognition_time = {}
        self.recognition_cooldown = 30  # seconds
        self.db_queue = Queue()
        
        # Load face detection cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Database setup
        self.setup_database()
        
        # Start database worker thread
        self.db_thread = threading.Thread(target=self.process_db_queue, daemon=True)
        self.db_thread.start()
        
        # Create GUI
        self.setup_gui()
        
        # Load data
        self.load_data()

    def process_frame(self, frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            
            # Compare with known faces
            if len(self.known_faces) > 0:
                # Simple face comparison using mean squared error
                min_error = float('inf')
                match_idx = -1
                
                for i, known_face in enumerate(self.known_faces):
                    error = np.mean((face - known_face) ** 2)
                    if error < min_error:
                        min_error = error
                        match_idx = i
                
                if min_error < 0.1:  # Threshold for face match
                    name = self.known_names[match_idx]
                    roll = self.known_rolls[match_idx]
                    
                    # Check cooldown
                    current_time = datetime.datetime.now()
                    if roll in self.last_recognition_time:
                        time_diff = (current_time - self.last_recognition_time[roll]).total_seconds()
                        if time_diff < self.recognition_cooldown:
                            continue
                    
                    # Update recognition time
                    self.last_recognition_time[roll] = current_time
                    
                    # Record attendance
                    self.record_attendance(name, roll)
                    
                    # Draw rectangle and name
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({roll})", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return frame

    def add_face(self, name, roll_number, face_image):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                raise Exception("No face detected in the image")
            
            if len(faces) > 1:
                raise Exception("Multiple faces detected in the image")
            
            # Extract and preprocess face
            x, y, w, h = faces[0]
            face = face_image[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            
            # Store face data
            self.known_faces.append(face)
            self.known_names.append(name)
            self.known_rolls.append(roll_number)
            
            # Store in database
            self.cursor.execute('''
                INSERT INTO students (name, roll_number, face_data)
                VALUES (?, ?, ?)
            ''', (name, roll_number, face.tobytes()))
            self.conn.commit()
            
            return True
            
        except Exception as e:
            print(f"Error adding face: {str(e)}")
            return False

    def load_data(self):
        try:
            self.cursor.execute('SELECT name, roll_number, face_data FROM students')
            rows = self.cursor.fetchall()
            
            self.known_faces = []
            self.known_names = []
            self.known_rolls = []
            
            for name, roll, face_data in rows:
                # Convert bytes to numpy array
                face = np.frombuffer(face_data, dtype=np.float32).reshape(64, 64, 3)
                
                self.known_faces.append(face)
                self.known_names.append(name)
                self.known_rolls.append(roll)
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")

    def record_attendance(self, name, roll_number):
        # Check if student is already checked in
        for record in self.attendance_data:
            if record['roll_number'] == roll_number and record['check_out'] is None:
                return  # Student already checked in
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.attendance_data.append({
            'name': name,
            'roll_number': roll_number,
            'check_in': current_time,
            'check_out': None
        })
        self.save_data()
        self.status_label.config(text=f"Status: Checked In - {name} ({roll_number})")
        self.refresh_lists()

    def setup_database(self):
        self.conn = sqlite3.connect('attendance.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                roll_number TEXT UNIQUE NOT NULL,
                face_data BLOB NOT NULL
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                check_in DATETIME,
                check_out DATETIME,
                date DATE,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        ''')
        
        self.conn.commit()
    
    def process_db_queue(self):
        while True:
            try:
                operation, data = self.db_queue.get()
                if operation == 'add_student':
                    name, roll_number, face_data = data
                    try:
                        self.cursor.execute('''
                            INSERT INTO students (name, roll_number, face_data)
                            VALUES (?, ?, ?)
                        ''', (name, roll_number, face_data))
                        self.conn.commit()
                    except sqlite3.IntegrityError:
                        messagebox.showerror("Error", "Roll number already exists!")
                        continue
                elif operation == 'check_in':
                    roll_number, check_in, date = data
                    self.cursor.execute('''
                        INSERT INTO attendance (student_id, check_in, date)
                        SELECT id, ?, ?
                        FROM students
                        WHERE roll_number = ?
                    ''', (check_in, date, roll_number))
                    self.conn.commit()
                elif operation == 'check_out':
                    roll_number, check_out = data
                    self.cursor.execute('''
                        UPDATE attendance
                        SET check_out = ?
                        WHERE student_id = (
                            SELECT id FROM students WHERE roll_number = ?
                        )
                        AND check_out IS NULL
                    ''', (check_out, roll_number))
                    self.conn.commit()
            except Exception as e:
                print(f"Database error: {e}")
            finally:
                self.db_queue.task_done()
    
    def setup_gui(self):
        # Create GUI with modern theme
        self.root = tk.Tk()
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("800x700")
        self.root.configure(bg='#1a1a1a')
        
        # Configure style
        style = ttk.Style()
        style.configure('TButton', padding=10, font=('Helvetica', 10))
        style.configure('TLabel', font=('Helvetica', 10), background='#1a1a1a', foreground='#00ff00')
        style.configure('TFrame', background='#1a1a1a')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, 
                              text="Face Recognition Attendance System",
                              font=('Helvetica', 20, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs
        self.attendance_tab = ttk.Frame(self.notebook)
        self.reports_tab = ttk.Frame(self.notebook)
        self.lists_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.attendance_tab, text="Attendance")
        self.notebook.add(self.reports_tab, text="Reports")
        self.notebook.add(self.lists_tab, text="Lists")
        
        # Attendance tab content
        self.setup_attendance_tab()
        
        # Reports tab content
        self.setup_reports_tab()
        
        # Lists tab content
        self.setup_lists_tab()
    
    def setup_attendance_tab(self):
        # Buttons frame
        button_frame = ttk.Frame(self.attendance_tab)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Create buttons with modern styling
        ttk.Button(button_frame, 
                  text="Add Student", 
                  command=self.add_student,
                  style='TButton').pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, 
                  text="Delete Student", 
                  command=self.delete_student,
                  style='TButton').pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, 
                  text="Start Auto Check-In", 
                  command=self.start_auto_check_in,
                  style='TButton').pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, 
                  text="Stop Auto Check-In", 
                  command=self.stop_auto_check_in,
                  style='TButton').pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, 
                  text="Start Auto Check-Out", 
                  command=self.start_auto_check_out,
                  style='TButton').pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, 
                  text="Stop Auto Check-Out", 
                  command=self.stop_auto_check_out,
                  style='TButton').pack(fill=tk.X, pady=5)
        
        # Status frame
        status_frame = ttk.Frame(self.attendance_tab)
        status_frame.pack(fill=tk.X, pady=20)
        
        # Status label with modern styling
        self.status_label = ttk.Label(status_frame, 
                                    text="Status: Ready",
                                    font=('Helvetica', 10))
        self.status_label.pack()
    
    def setup_reports_tab(self):
        # Calendar frame
        calendar_frame = ttk.Frame(self.reports_tab)
        calendar_frame.pack(fill=tk.X, pady=10)
        
        # Calendar widget
        self.calendar = DateEntry(calendar_frame, 
                               date_pattern='yyyy-mm-dd',
                               background='#1a1a1a',
                               foreground='#00ff00',
                               selectbackground='#00ff00',
                               selectforeground='#1a1a1a')
        self.calendar.pack(pady=10)
        
        # Report buttons
        ttk.Button(calendar_frame,
                  text="Generate Daily Report",
                  command=self.generate_daily_report,
                  style='TButton').pack(pady=5)
        
        ttk.Button(calendar_frame,
                  text="Generate Monthly Report",
                  command=self.generate_monthly_report,
                  style='TButton').pack(pady=5)
        
        # Report display area
        self.report_text = tk.Text(self.reports_tab,
                                 height=20,
                                 width=70,
                                 bg='#1a1a1a',
                                 fg='#00ff00',
                                 font=('Courier', 10))
        self.report_text.pack(pady=10)
    
    def setup_lists_tab(self):
        # Create frames for each list
        all_students_frame = ttk.LabelFrame(self.lists_tab, text="All Students", padding="10")
        all_students_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        checked_in_frame = ttk.LabelFrame(self.lists_tab, text="Currently Checked In", padding="10")
        checked_in_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        checked_out_frame = ttk.LabelFrame(self.lists_tab, text="Today's Checked Out", padding="10")
        checked_out_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create text widgets for each list
        self.all_students_text = tk.Text(all_students_frame,
                                       height=8,
                                       width=70,
                                       bg='#1a1a1a',
                                       fg='#00ff00',
                                       font=('Courier', 10))
        self.all_students_text.pack(fill=tk.BOTH, expand=True)
        
        self.checked_in_text = tk.Text(checked_in_frame,
                                     height=8,
                                     width=70,
                                     bg='#1a1a1a',
                                     fg='#00ff00',
                                     font=('Courier', 10))
        self.checked_in_text.pack(fill=tk.BOTH, expand=True)
        
        self.checked_out_text = tk.Text(checked_out_frame,
                                      height=8,
                                      width=70,
                                      bg='#1a1a1a',
                                      fg='#00ff00',
                                      font=('Courier', 10))
        self.checked_out_text.pack(fill=tk.BOTH, expand=True)
        
        # Add refresh button
        ttk.Button(self.lists_tab,
                  text="Refresh Lists",
                  command=self.refresh_lists,
                  style='TButton').pack(pady=10)
    
    def generate_daily_report(self):
        selected_date = self.calendar.get_date()
        self.cursor.execute('''
            SELECT s.name, s.roll_number, a.check_in, a.check_out
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE date(a.date) = ?
            ORDER BY a.check_in
        ''', (selected_date,))
        
        records = self.cursor.fetchall()
        self.display_report(records, f"Daily Report for {selected_date}")
    
    def generate_monthly_report(self):
        selected_date = self.calendar.get_date()
        year, month = selected_date.split('/')[:2]
        
        self.cursor.execute('''
            SELECT s.name, s.roll_number,
                   COUNT(DISTINCT date(a.date)) as days_present,
                   AVG(strftime('%s', a.check_out) - strftime('%s', a.check_in))/3600 as avg_hours
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE strftime('%Y-%m', a.date) = ?
            GROUP BY s.id
            ORDER BY days_present DESC
        ''', (f"{year}-{month}",))
        
        records = self.cursor.fetchall()
        self.display_report(records, f"Monthly Report for {year}-{month}")
    
    def display_report(self, records, title):
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, f"{title}\n{'='*50}\n\n")
        
        for record in records:
            if len(record) == 4:  # Daily report
                name, roll, check_in, check_out = record
                self.report_text.insert(tk.END, 
                    f"Name: {name}\nRoll: {roll}\nCheck-in: {check_in}\nCheck-out: {check_out}\n{'-'*30}\n")
            else:  # Monthly report
                name, roll, days, hours = record
                self.report_text.insert(tk.END, 
                    f"Name: {name}\nRoll: {roll}\nDays Present: {days}\nAvg Hours: {hours:.2f}\n{'-'*30}\n")
    
    def save_data(self):
        current_time = datetime.datetime.now()
        for record in self.attendance_data:
            if record['check_out'] is None:
                # Check if this check-in is already saved
                self.cursor.execute('''
                    SELECT id FROM attendance 
                    WHERE student_id = (SELECT id FROM students WHERE roll_number = ?)
                    AND date = ? AND check_in = ?
                ''', (record['roll_number'], current_time.date(), record['check_in']))
                
                if not self.cursor.fetchone():
                    self.db_queue.put(('check_in', (record['roll_number'], record['check_in'], current_time.date())))
            else:
                # Check if this check-out is already saved
                self.cursor.execute('''
                    SELECT id FROM attendance 
                    WHERE student_id = (SELECT id FROM students WHERE roll_number = ?)
                    AND check_in = ? AND check_out IS NULL
                ''', (record['roll_number'], record['check_in']))
                
                if self.cursor.fetchone():
                    self.db_queue.put(('check_out', (record['roll_number'], record['check_out'])))
    
    def add_student(self):
        # Create a new window for student registration
        reg_window = tk.Toplevel(self.root)
        reg_window.title("Add New Student")
        reg_window.geometry("400x300")
        reg_window.configure(bg='#1a1a1a')
        
        # Center the window
        reg_window.transient(self.root)
        reg_window.grab_set()
        
        # Create frames
        input_frame = ttk.Frame(reg_window, padding="20")
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name entry
        ttk.Label(input_frame, text="Enter student name:").pack(pady=5)
        name_entry = ttk.Entry(input_frame, width=30)
        name_entry.pack(pady=5)
        
        # Roll number entry
        ttk.Label(input_frame, text="Enter roll number:").pack(pady=5)
        roll_entry = ttk.Entry(input_frame, width=30)
        roll_entry.pack(pady=5)
        
        def capture_and_save():
            name = name_entry.get().strip()
            roll_number = roll_entry.get().strip()
            
            if not name or not roll_number:
                messagebox.showerror("Error", "Please fill in all fields")
                return
            
            # Check if roll number already exists
            self.cursor.execute('SELECT id FROM students WHERE roll_number = ?', (roll_number,))
            if self.cursor.fetchone():
                messagebox.showerror("Error", "Roll number already exists!")
                return
            
            reg_window.destroy()
            messagebox.showinfo("Instructions", "Look at the camera and press SPACE to capture your face")
            
            # Capture face
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Show frame
                cv2.imshow('Capture Face - Press SPACE to capture', frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space pressed
                    face_encoding = self.preprocess_face(frame)
                    if face_encoding is not None:
                        # Add to database
                        try:
                            self.cursor.execute('''
                                INSERT INTO students (name, roll_number, face_data)
                                VALUES (?, ?, ?)
                            ''', (name, roll_number, face_encoding.tobytes()))
                            self.conn.commit()
                            
                            # Update local data
                            self.known_faces.append(face_encoding)
                            self.known_names.append(name)
                            self.known_rolls.append(roll_number)
                            
                            messagebox.showinfo("Success", "Student added successfully!")
                            self.refresh_lists()
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to add student: {str(e)}")
                    else:
                        messagebox.showerror("Error", "No face detected. Please try again.")
                    break
                elif key == ord('q'):  # Q pressed
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        # Capture button
        ttk.Button(input_frame, 
                  text="Capture Face", 
                  command=capture_and_save,
                  style='TButton').pack(pady=20)
    
    def preprocess_face(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Extract and preprocess face
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            return face
        return None

    def recognize_face(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None, None
        
        # Extract and preprocess face
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        
        # Compare with known faces using vectorized operations
        if len(self.known_faces) > 0:
            # Convert face to flat array
            face_flat = face.reshape(-1)
            # Calculate differences for all faces at once
            differences = np.array([np.mean((face_flat - known_face.reshape(-1)) ** 2) 
                                 for known_face in self.known_faces])
            
            min_idx = np.argmin(differences)
            min_diff = differences[min_idx]
            
            if min_diff < 0.1:  # Threshold for face match
                return self.known_names[min_idx], self.known_rolls[min_idx]
        
        return None, None

    def auto_detect_loop(self, mode='check_in'):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Smaller frame size
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # for faster processing
        
        face_detection_interval = 0  # Frame counter
        detected_face = None
        detected_name = None
        detected_roll = None
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process every 3rd frame for face detection
            if face_detection_interval % 3 == 0:
                # Process frame
                name, roll_number = self.recognize_face(frame)
                if name and roll_number:
                    detected_face = frame
                    detected_name = name
                    detected_roll = roll_number
            
            # Draw rectangle and name if face was detected
            if detected_face is not None:
                current_time = datetime.datetime.now()
                # Check cooldown period
                if detected_roll not in self.last_recognition_time or \
                   (current_time - self.last_recognition_time[detected_roll]).total_seconds() >= self.recognition_cooldown:
                    
                    if mode == 'check_in':
                        self.process_check_in(detected_name, detected_roll)
                    else:
                        self.process_check_out(detected_name, detected_roll)
                    
                    self.last_recognition_time[detected_roll] = current_time
                    detected_face = None  # Reset detection
            
            face_detection_interval += 1
            if face_detection_interval > 100:  # Reset counter
                face_detection_interval = 0
            
            # Display the frame
            cv2.imshow('Auto Detection - Press Q to stop', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_check_in(self, name, roll_number):
        # Check if student is already checked in
        for record in self.attendance_data:
            if record['name'] == name and record['roll_number'] == roll_number and record['check_out'] is None:
                messagebox.showwarning("Warning", f"{name} is already checked in!")
                return

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.attendance_data.append({
            'name': name,
            'roll_number': roll_number,
            'check_in': current_time,
            'check_out': None
        })
        self.save_data()
        self.status_label.config(text=f"Status: Checked In - {name} ({roll_number})")
        messagebox.showinfo("Success", f"{name} has been checked in successfully!")
        
        self.refresh_lists()  # Refresh lists after check-in
    
    def process_check_out(self, name, roll_number):
        try:
            # Use direct database query for faster check-out
            self.cursor.execute('''
                UPDATE attendance 
                SET check_out = ? 
                WHERE student_id = (
                    SELECT id FROM students WHERE roll_number = ?
                ) 
                AND check_out IS NULL
                AND date = ?
            ''', (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                 roll_number, 
                 datetime.datetime.now().date()))
            
            if self.cursor.rowcount > 0:
                self.conn.commit()
                self.status_label.config(text=f"Status: Checked Out - {name} ({roll_number})")
                # Update local data
                for record in self.attendance_data:
                    if record['roll_number'] == roll_number and record['check_out'] is None:
                        record['check_out'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        break
                self.refresh_lists()
            else:
                messagebox.showwarning("Warning", f"{name} is not checked in!")
                
        except Exception as e:
            print(f"Check-out error: {str(e)}")
            messagebox.showerror("Error", "Failed to process check-out")
    
    def start_auto_check_in(self):
        if not self.known_faces:
            messagebox.showerror("Error", "Please add students first!")
            return
        self.is_running = True
        threading.Thread(target=self.auto_detect_loop, args=('check_in',)).start()
    
    def start_auto_check_out(self):
        if not self.known_faces:
            messagebox.showerror("Error", "Please add students first!")
            return
        self.is_running = True
        threading.Thread(target=self.auto_detect_loop, args=('check_out',)).start()
    
    def stop_auto_check_in(self):
        self.is_running = False
        self.status_label.config(text="Status: Auto Check-In Stopped")
    
    def stop_auto_check_out(self):
        self.is_running = False
        self.status_label.config(text="Status: Auto Check-Out Stopped")
    
    def quit_application(self):
        self.is_running = False
        self.conn.close()
        self.root.quit()
    
    def run(self):
        self.root.mainloop()

    def refresh_lists(self):
        # Clear all text widgets
        self.all_students_text.delete(1.0, tk.END)
        self.checked_in_text.delete(1.0, tk.END)
        self.checked_out_text.delete(1.0, tk.END)
        
        # Get all students
        self.cursor.execute('SELECT name, roll_number FROM students ORDER BY name')
        all_students = self.cursor.fetchall()
        
        # Display all students
        self.all_students_text.insert(tk.END, "Name\t\tRoll Number\n")
        self.all_students_text.insert(tk.END, "-" * 50 + "\n")
        for name, roll in all_students:
            self.all_students_text.insert(tk.END, f"{name}\t\t{roll}\n")
        
        # Get currently checked in students
        today = datetime.datetime.now().date()
        self.cursor.execute('''
            SELECT s.name, s.roll_number, a.check_in
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE date(a.date) = ? AND a.check_out IS NULL
            ORDER BY a.check_in
        ''', (today,))
        checked_in = self.cursor.fetchall()
        
        # Display checked in students
        self.checked_in_text.insert(tk.END, "Name\t\tRoll Number\tCheck-in Time\n")
        self.checked_in_text.insert(tk.END, "-" * 70 + "\n")
        for name, roll, check_in in checked_in:
            self.checked_in_text.insert(tk.END, f"{name}\t\t{roll}\t{check_in}\n")
        
        # Get today's checked out students
        self.cursor.execute('''
            SELECT s.name, s.roll_number, a.check_in, a.check_out
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE date(a.date) = ? AND a.check_out IS NOT NULL
            ORDER BY a.check_out DESC
        ''', (today,))
        checked_out = self.cursor.fetchall()
        
        # Display checked out students
        self.checked_out_text.insert(tk.END, "Name\t\tRoll Number\tCheck-in\tCheck-out\n")
        self.checked_out_text.insert(tk.END, "-" * 90 + "\n")
        for name, roll, check_in, check_out in checked_out:
            self.checked_out_text.insert(tk.END, f"{name}\t\t{roll}\t{check_in}\t{check_out}\n")

    def delete_student(self):
        # Create a new window for student deletion
        del_window = tk.Toplevel(self.root)
        del_window.title("Delete Student")
        del_window.geometry("400x300")
        del_window.configure(bg='#1a1a1a')
        
        # Center the window
        del_window.transient(self.root)
        del_window.grab_set()
        
        # Create frames
        input_frame = ttk.Frame(del_window, padding="20")
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Get list of students
        self.cursor.execute('SELECT name, roll_number FROM students ORDER BY name')
        students = self.cursor.fetchall()
        
        if not students:
            messagebox.showinfo("Info", "No students registered in the system.")
            del_window.destroy()
            return
        
        # Create combobox for student selection
        ttk.Label(input_frame, text="Select student to delete:").pack(pady=5)
        student_var = tk.StringVar()
        student_combo = ttk.Combobox(input_frame, 
                                   textvariable=student_var,
                                   values=[f"{name} ({roll})" for name, roll in students],
                                   state="readonly",
                                   width=30)
        student_combo.pack(pady=5)
        
        def confirm_delete():
            selected = student_var.get()
            if not selected:
                messagebox.showerror("Error", "Please select a student to delete")
                return
            
            # Extract roll number from selection
            roll_number = selected.split('(')[1].strip(')')
            
            # Confirm deletion
            if messagebox.askyesno("Confirm", "Are you sure you want to delete this student?"):
                try:
                    # Delete student from database
                    self.cursor.execute('DELETE FROM students WHERE roll_number = ?', (roll_number,))
                    self.conn.commit()
                    
                    # Update local data
                    idx = self.known_rolls.index(roll_number)
                    self.known_faces.pop(idx)
                    self.known_names.pop(idx)
                    self.known_rolls.pop(idx)
                    
                    messagebox.showinfo("Success", "Student deleted successfully!")
                    del_window.destroy()
                    self.refresh_lists()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete student: {str(e)}")
        
        # Delete button
        ttk.Button(input_frame, 
                  text="Delete Student", 
                  command=confirm_delete,
                  style='TButton').pack(pady=20)

if __name__ == "__main__":
    app = FaceRecognitionAttendance()
    app.run() 