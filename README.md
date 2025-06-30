# soccer-multi-object-tracking
Soccer Player &amp; Ball Multi-Object Tracking with YOLOv8 + DeepSORT, ByteTrack, and BoT-SORT. Supports team assignment, re-identification, and occlusion recovery with plug-and-play modular architecture.


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
   Place your fine-tuned YOLO model (`best.pt`) inside the newly created `models` folder:

   ```bash
   mkdir models
   # then move best.pt into this folder
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

6. **Verify the model placement**
   Make sure that `models/best.pt` exists.

7. **Run the main script**

   ```bash
   python main.py
   ``` 


   You will see three options to select from. Choose one, and wait for a few minutes for processing to complete (execution time depends on your system’s computational power — using a GPU is highly recommended).

