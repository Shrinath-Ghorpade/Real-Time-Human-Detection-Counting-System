#  Real-Time Human Detection & Counting System  

This project is a **Flask-based web application** that detects and counts humans in **images, videos, and live camera feed** using **OpenCV’s HOG + SVM** algorithm.  

---

##  Features  
-  Upload an image and detect humans.  
-  Upload a video and detect humans frame by frame.  
-  Open camera for real-time human detection & counting.  
-  Web interface with Flask (HTML + CSS).  
-  Processed results are displayed with bounding boxes & counts.  

---

## Tech Stack  
- **Backend**: Python, Flask  
- **Computer Vision**: OpenCV (HOG + SVM), imutils  
- **Frontend**: HTML, CSS  
- **Deployment**: GitHub / Any cloud (Heroku, Render, etc.)  

---

##  Project Structure  
  
```
RealTime-Human-Detection/
│── app.py              # Flask application  
│── detect.py           # Human detection logic  
│── requirements.txt    # Dependencies  
│── .gitignore          # Ignore unnecessary files  
│
├── static/             # Static files  
│   ├── style.css  
│   ├── uploads/        # Stores uploaded images/videos  
│   └── results/        # Stores processed results  
│
├── templates/          # HTML templates  
│   ├── index.html      # Upload page  
│   └── result.html     # Result display page  
```

##  Installation
1. **Clone the repo**  
   ```bash
   git clone https://github.com/YOUR_USERNAME/RealTime-Human-Detection.git
   cd RealTime-Human-Detection
2.  **Create virtual environment**
     ```bash

    python -m venv .venv
    .venv\Scripts\activate        # On Windows
    source .venv/bin/activate     # On Linux/Mac
     ```
3.  **Install dependencies**
    ```bash
    
    pip install -r requirements.txt
    ```

4.  **Run the Flask app**
    ```bash
    python app.py

