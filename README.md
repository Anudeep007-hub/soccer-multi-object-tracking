

# ⚽ soccer-multi-object-tracking

Soccer Player & Ball Multi-Object Tracking with YOLOv8 + DeepSORT, ByteTrack, and BoT-SORT. Supports team assignment, player re-identification, and occlusion recovery with a plug-and-play modular architecture.

This project enables consistent player and ball tracking in soccer match videos, with robust ID assignment even across re-entries, partial occlusions, and crowded scenes.

---

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <YOUR_REPO_URL>  # use HTTPS, SSH, or CLI based on your preference
   ```

2. **Navigate to the project directory**

   ```bash
   cd <project-folder>
   ```

3. **Create the `models` folder**
    **Important**: You *must* create a `models` folder and place your fine-tuned YOLO model file (`best.pt`) inside it. Otherwise the tracker will fail to load.

   ```bash
   mkdir models
   # then move your fine-tuned best.pt into this folder
   ```

   **Make sure this path is correct**:

   ```
   models/best.pt
   ```

4. **Set up a virtual environment and activate it**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   *(or use your preferred environment manager)*

5. **Install the dependencies**

   ```bash
   pip install -r requirements.txt
   ```

6. **Verify model placement**
   Check that you have:

   ```
   models/best.pt
   ```

   before running, or you will see a model loading error.

7. **Run the main script**

   ```bash
   python main.py
   ```

   You will see three options to select from. Choose one, and wait for a few minutes for processing to complete (execution time depends on your system’s computational power — using a GPU is highly recommended).

---

# 🎥 Demo Videos

## 🎥 Demo Videos

### 📌 Initial Video

> *This is the raw input video with **distracting** bounding boxes and no proper tracking yet:*

[▶️ Watch Initial Video](https://drive.google.com/file/d/12ur9PlVxo_9kyG-NAL7uQBZBOrPYdUmD/view?usp=sharing)

---

### 📌 YOLO + ByteTrack

[▶️ Watch YOLO + ByteTrack](https://drive.google.com/file/d/1IHcE9agV7A3BKTZphJFt7t4jD3T_xZRj/view?usp=sharing)  
**Run time:** ~4.78 sec

---

### 📌 YOLO + DeepSORT (MobileNet)

[▶️ Watch YOLO + DeepSORT (MobileNet)](https://drive.google.com/file/d/1xtu_T6fFWL6O-slPQWbFYz5xGHkE8qTG/view?usp=sharing)  
**Run time:** ~4.86 sec

---

### 📌 YOLO + BoTSORT

> *This video uses different bounding box styles compared to the above methods:*

[▶️ Watch YOLO + BoTSORT](https://drive.google.com/file/d/12ur9PlVxo_9kyG-NAL7uQBZBOrPYdUmD/view?usp=sharing)  
**Run time:** ~28.28 sec

> **Note**: All these video files are also included in the `output_videos` folder of this repository for local viewing or offline playback.

---

# 📝 Project Details

* **Object Detection Backbone**: YOLOv8 fine-tuned on soccer players
* **Tracking Algorithms**:

  * ByteTrack (for strong identity preservation under occlusion)
  * DeepSORT with MobileNet re-identification features
  * BoT-SORT (inbuilt tracker with tuned hyperparameters)
* **Modularity**: The architecture supports easy switching between trackers.
* **Efficiency**: Designed for near real-time processing on standard GPUs.

---

# IMPORTANT NOTE

If you are reviewing this project, please note:

* Pese create a folder "models" after cloning.
* Model file (`best.pt`) must be placed inside `models/`
* Video results shown above are provided for quick comparison
* Code is organized modularly, with clear separation of detection and tracking
* Performance may vary based on hardware (tests were run on an RTX 4060 8GB VRAM)

