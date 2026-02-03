# Real-Time People Flow Analytics using Computer Vision

##  Overview
This project implements a **real-time people flow and occupancy analytics system** using computer vision and deep learning. The system detects and tracks people in live video streams and counts entry and exit events using line-crossing logic. Each individual is assigned a unique tracking ID to prevent duplicate counting across frames.

The solution is suitable for applications such as **footfall analysis, crowd monitoring, entry–exit management, and occupancy tracking** in environments like offices, malls, campuses, libraries, and public spaces.

---

##  Key Features
- Real-time **person detection** using YOLO
- **Multi-object tracking** with persistent IDs
- Entry and exit counting using **line-crossing logic**
- Prevents duplicate counting of the same individual
- Live **FPS monitoring**
- Supports **webcam and CCTV-style video feeds**
- Logs people flow data for further analysis

---

##  Technologies Used
- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- ByteTrack (Multi-object tracking)  

---

---

##  How It Works
1. The system captures frames from a live video source (webcam or CCTV feed).
2. A YOLO-based model detects people in each frame.
3. A tracking algorithm assigns unique IDs to detected individuals.
4. When a tracked person crosses a predefined line, an **entry or exit event** is registered.
5. The system ensures each person is counted only once per crossing event.

---

##  Running the Project
1. Ensure Python is installed.
2. Install required dependencies:
   - `ultralytics`
   - `opencv-python`
3. Run the script:
   ```bash
   python detect.py


## 📂 Project Structure
