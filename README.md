# SMART GARBAGE CLASSIFICATION SYSTEM USING ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING

> A complete, production-grade, full-stack AI/ML web application designed to automatically identify, classify, and guide the segregation of municipal solid waste into seven standardized categories in real-time.

---

## 1. Project Objective

Urbanization and consumer consumption generate billions of tons of solid waste annually. A large portion of recyclable and compostable waste ends up in landfills because of improper waste segregation at the source.

The **Smart Garbage Classification System using Artificial Intelligence and Machine Learning** provides an automated, accessible, computer-vision solution. By scanning or capturing images of garbage items, the system utilizes Deep Convolutional Neural Networks (CNN) and Machine Learning models to:
- Instantly identify the waste category.
- Calculate and display neural network prediction confidence percentages.
- Deliver scientific, actionable disposal and recycling recommendations.
- Distinguish whether materials are recyclable, biodegradable, or non-recyclable.
- Record classifications in a persistent database to provide analytical dashboards and downloadable audit reports.

---

## 2. Waste Classification Categories

The system classifies garbage into **7 standard waste streams**:

| Category | Typical Items | Recyclable? | Biodegradable? | Disposal Recommendation |
| :--- | :--- | :---: | :---: | :--- |
| **1. Organic Waste** | Food leftovers, vegetable peels, fruit rinds, leaves, coffee grounds | ❌ | ✅ | Compost at home or dispose in municipal green organic bins for bio-fertilizer. |
| **2. Plastic Waste** | PET bottles, covers, food containers, caps, wrappers | ✅ | ❌ | Empty contents, rinse clean, flatten, and deposit in blue recycling bins. |
| **3. Paper Waste** | Newspapers, cardboard shipping boxes, magazines, notebooks | ✅ | ✅ | Keep clean and dry. Flatten cardboard cartons and bundle for paper pulping. |
| **4. Metal Waste** | Aluminium soda cans, tin cans, iron scraps, foil trays | ✅ | ❌ | Rinse food residue and drop off at scrap metal dealers or yellow recycling bins. |
| **5. Glass Waste** | Glass beverage bottles, sauce jars, cosmetic bottles, glass fragments | ✅ | ❌ | Rinse clean. Wrap broken glass securely to protect handlers before recycling. |
| **6. E-Waste** | Lithium batteries, phone chargers, cables, circuit boards, chips | ✅ | ❌ | Never throw in household trash. Hand over to certified e-waste collection kiosks. |
| **7. General Waste** | Non-recyclable ceramics, sanitary waste, multilayer composite foil | ❌ | ❌ | Tie securely in trash bags and place in black/grey municipal landfill bins. |

---

## 3. Technology Stack

### Frontend:
- **HTML5**: Semantic tags, accessible forms, audio/video elements.
- **CSS3**: Custom environmental design system, smooth glassmorphism, responsive breakpoints.
- **JavaScript (ES6+)**: Asynchronous fetch API, webcam MediaDevices stream handler, canvas image rendering.
- **Bootstrap 5.3**: Responsive UI components, cards, grid layout, modal dialogs.
- **Chart.js 4.4**: Interactive dynamic Doughnut, Bar, and Line charts.
- **Bootstrap Icons 1.11**: Visual cues and symbols across all pages.

### Backend:
- **Python 3**: Core application runtime.
- **Flask**: Lightweight, high-performance web framework.
- **Werkzeug**: Secure filename validation and multipart request parsing.

### AI / Machine Learning:
- **TensorFlow / Keras 3.x**: Convolutional Neural Network (CNN) architecture with Conv2D, BatchNormalization, MaxPooling2D, and Softmax layers.
- **Scikit-Learn**: High-performance Deep Multi-Layer Neural Network Vision pipeline (`MLPClassifier`, `StandardScaler`).
- **Pillow (PIL)**: High-resolution image resampling, 224x224 tensor normalization, spatial grid analysis, and edge detection.
- **NumPy**: Multi-dimensional matrix operations and probability vector scaling.

### Database:
- **SQLite3**: Lightweight, zero-configuration database storing classification logs, timestamps, confidence scores, and recycling metadata.

---

## 4. Key Website Pages & Features

### A. Home Page (`/`)
- Environmental hero section with gradient animations and real-time engine status.
- Global environmental awareness statistics (landfill metrics, energy savings).
- Step-by-step visual workflow (Upload -> CNN Inference -> Segregate).
- Interactive cards for all 7 waste categories with material badges.

### B. Garbage Classification Page (`/classify`)
- **Dual-Mode Input**:
  1. **Upload Tab**: Drag-and-drop zone with instant image preview, clear button, and client-side validation (< 16MB).
  2. **Live Webcam Tab**: Live video streaming with target alignment guide, frame capture, snapshot freezing, and camera toggle.
- **One-Click Sample Images**: Instant testing buttons for Apple, Water Bottle, Cardboard, Soda Can, Glass Bottle, and Battery.
- **Live AI Prediction Card**:
  - Predicted waste category with theme-matched color banner and icon.
  - Neural network confidence score badge.
  - Recyclable vs Biodegradable status pills.
  - Smart disposal guidelines and eco-action tips.
  - Probability distribution breakdown across all 7 categories with progress bars.

### C. Analytics Dashboard (`/dashboard`)
- KPI Summary Cards: Total items classified, Recyclable %, Biodegradable %, Average confidence score.
- 7-Category Frequency Badges.
- Interactive **Doughnut / Pie Chart** showing proportional distribution.
- Interactive **Bar Chart** illustrating item counts per waste stream.
- Recent classification history log table with image thumbnails.

### D. History Log (`/history`)
- Comprehensive audit table of all historical classifications.
- Instant search filter by filename or disposal recommendation.
- Category filter dropdown.
- Individual record deletion with modal confirmation.
- Clear All History button.
- **CSV Report Export** button.

### E. About Page (`/about`)
- Mission overview and municipal waste statistics.
- CNN Neural Network architectural diagram and layer description.
- Complete breakdown of frontend, backend, AI/ML, and database technologies.
- Environmental impact insights (methane reduction, circular economy).

---

## 5. Project Folder Structure

```
smart-garbage-classification system/
│
├── app.py                     # Main Flask web server & API controller
├── train_model.py             # AI/ML CNN & Neural Network training pipeline
├── model.py                   # Model inference, image preprocessing (224x224), metadata
├── database.py                # SQLite database operations & analytics queries
├── create_samples.py          # Sample test images generator
├── seed_database.py           # Initial database seeder
├── requirements.txt           # Python dependencies
├── run_app.bat                # Windows double-click application launcher
├── train.bat                  # Windows double-click model training launcher
├── README.md                  # Comprehensive documentation
│
├── dataset/                   # Garbage dataset split into train/ and test/
│   ├── train/
│   │   ├── organic_waste/
│   │   ├── plastic_waste/
│   │   ├── paper_waste/
│   │   ├── metal_waste/
│   │   ├── glass_waste/
│   │   ├── e_waste/
│   │   └── general_waste/
│   └── test/
│       └── ...
│
├── models/                    # Trained model checkpoints
│   ├── garbage_classifier_model.joblib
│   ├── garbage_classifier_classes.json
│   └── training_metrics.json
│
├── uploads/                   # Stored user-uploaded and captured images
├── sample_images/             # Pre-configured demonstration images
│
├── templates/                 # Jinja2 HTML templates
│   ├── base.html              # Base layout with navbar & footer
│   ├── index.html             # Landing home page
│   ├── classify.html          # File upload & live camera classification
│   ├── dashboard.html         # Real-time analytics charts
│   ├── history.html           # Historical records table & search
│   └── about.html             # Project details & CNN architecture
│
└── static/
    ├── css/
    │   └── style.css          # Environmental green & white custom CSS
    └── js/
        └── script.js          # Webcam capture, drag-and-drop, AJAX requests
```

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13 installed.
- Modern web browser (Chrome, Edge, Firefox, Safari).

### Step 1: Install Dependencies
Open PowerShell or Command Prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Train or Re-train the AI Model
To train the neural network on the dataset and display training & validation accuracy:
```bash
python train_model.py
```
*(On Windows, you can simply double-click `train.bat`)*

### Step 3: Run the Web Application
Start the Flask web server:
```bash
python app.py
```
*(On Windows, you can simply double-click `run_app.bat`)*

### Step 4: Open in Web Browser
Navigate to:
```
http://127.0.0.1:5000
```

---

## 7. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Home Landing Page |
| `GET` | `/classify` | Classification & Live Camera Page |
| `POST` | `/api/classify` | AI inference endpoint (Multipart file, base64 camera image, or sample filename) |
| `GET` | `/dashboard` | Analytics Dashboard Page |
| `GET` | `/api/dashboard-data` | JSON statistics for asynchronous Chart.js rendering |
| `GET` | `/history` | Historical Classifications Page with search & category filter |
| `POST` | `/api/history/delete/<id>` | Delete a specific classification record and uploaded image |
| `POST` | `/api/history/clear` | Clear all historical records |
| `GET` | `/export-csv` | Download complete classification history as CSV |
| `GET` | `/about` | About the System & Neural Network Architecture |

---

## 8. Environmental Impact

- **Methane Abatement**: Diverting organic matter from landfills into composting eliminates anaerobic decay that produces methane (a greenhouse gas 28× more potent than carbon dioxide).
- **Resource Recovery**: Pure recycling streams for plastic, aluminium, and glass allow continuous re-melting, preserving precious fossil fuel and mining resources.
- **Toxin Mitigation**: Intercepting electronic waste keeps heavy metals such as lead, cadmium, and mercury out of groundwater tables.

---

## 9. License & Credits

Built for academic, educational, and sustainable smart city deployment.
Licensed under the MIT License.
