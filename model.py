"""
AI/ML Model Management and Inference Engine
Handles image preprocessing, model loading, and garbage classification predictions.
"""

import os
import json
import numpy as np
from PIL import Image
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
KERAS_MODEL_PATH = os.path.join(MODELS_DIR, 'garbage_classifier.keras')
NN_MODEL_PATH = os.path.join(MODELS_DIR, 'garbage_classifier_model.joblib')
CLASSES_PATH = os.path.join(MODELS_DIR, 'garbage_classifier_classes.json')

# Standard 7 waste categories matching requirements
ORDERED_CLASSES = [
    "Organic Waste",
    "Plastic Waste",
    "Paper Waste",
    "Metal Waste",
    "Glass Waste",
    "E-Waste",
    "General Waste"
]

WASTE_METADATA = {
    "Organic Waste": {
        "category": "Organic Waste",
        "title": "Organic / Biodegradable Waste",
        "description": "Food scraps, vegetable peels, fruit rinds, leaves, tea bags, and organic kitchen leftovers.",
        "examples": "Vegetable peels, fruit remains, leftover food, garden leaves, egg shells, coffee grounds",
        "recommendation": "Compost at home or place into the green organic waste bin for municipal compost & bio-fertilizer.",
        "is_recyclable": False,
        "is_biodegradable": True,
        "color": "#198754",
        "badge_class": "bg-success",
        "icon": "bi-flower1",
        "eco_tip": "Keep organic waste separated from plastic wrappers to avoid microplastic contamination in compost."
    },
    "Plastic Waste": {
        "category": "Plastic Waste",
        "title": "Recyclable Plastic Waste",
        "description": "Synthetic polymers including PET drink bottles, HDPE milk jugs, covers, and containers.",
        "examples": "Plastic bottles, poly bags, food containers, caps, snack wrappers, shampoo bottles",
        "recommendation": "Empty contents, rinse clean, flatten, and deposit into designated blue plastic recycling bins.",
        "is_recyclable": True,
        "is_biodegradable": False,
        "color": "#0d6efd",
        "badge_class": "bg-primary",
        "icon": "bi-recycle",
        "eco_tip": "Rinsing plastic prevents odor and stops batches from being rejected at recycling facilities."
    },
    "Paper Waste": {
        "category": "Paper Waste",
        "title": "Paper & Cardboard Waste",
        "description": "Cellulose fiber products such as newspapers, cardboard shipping boxes, envelopes, and notebooks.",
        "examples": "Cardboard cartons, newspapers, magazines, office documents, paper bags, egg cartons",
        "recommendation": "Keep clean and dry. Flatten cardboard cartons and bundle newspapers for paper pulp recycling.",
        "is_recyclable": True,
        "is_biodegradable": True,
        "color": "#ffc107",
        "badge_class": "bg-warning text-dark",
        "icon": "bi-journal-text",
        "eco_tip": "Do not mix greasy food-stained paper (like greasy pizza boxes) with clean paper recycling."
    },
    "Metal Waste": {
        "category": "Metal Waste",
        "title": "Metal & Can Waste",
        "description": "Ferrous and non-ferrous metals including beverage cans, tin food cans, foil, and scrap metal.",
        "examples": "Aluminium beverage cans, tin food cans, foil trays, metal bottle caps, iron scraps",
        "recommendation": "Rinse to remove food residue and drop off in metal recycling bins or local scrap collection.",
        "is_recyclable": True,
        "is_biodegradable": False,
        "color": "#6c757d",
        "badge_class": "bg-secondary",
        "icon": "bi-shield-check",
        "eco_tip": "Aluminium can be recycled infinitely with up to 95% energy savings compared to raw ore extraction."
    },
    "Glass Waste": {
        "category": "Glass Waste",
        "title": "Glass Containers & Jars",
        "description": "Silica glassware, condiment jars, beverage bottles, cosmetic bottles, and glass fragments.",
        "examples": "Glass beverage bottles, sauce jars, jam pots, perfume containers, glass shards",
        "recommendation": "Rinse containers and place in glass collection points. Wrap broken glass safely before disposal.",
        "is_recyclable": True,
        "is_biodegradable": False,
        "color": "#0dcaf0",
        "badge_class": "bg-info text-dark",
        "icon": "bi-cup-straw",
        "eco_tip": "Glass never degrades in chemical purity and can be melted and remolded into new bottles endlessly."
    },
    "E-Waste": {
        "category": "E-Waste",
        "title": "Electronic & Electrical Waste",
        "description": "End-of-life electronics, consumer appliances, circuit boards, batteries, and cables.",
        "examples": "Batteries, smartphone chargers, power cables, computer parts, circuit boards, old remotes",
        "recommendation": "Do not throw into household trash. Hand over to certified e-waste recyclers or retail take-back bins.",
        "is_recyclable": True,
        "is_biodegradable": False,
        "color": "#dc3545",
        "badge_class": "bg-danger",
        "icon": "bi-battery-charging",
        "eco_tip": "E-waste contains hazardous materials like mercury and lead, but also valuable recoverable gold and copper."
    },
    "General Waste": {
        "category": "General Waste",
        "title": "General / Non-Recyclable Waste",
        "description": "Mixed, composite, or contaminated materials that cannot be recycled through standard facilities.",
        "examples": "Ceramic pieces, sanitary hygiene items, vacuum dust, multilayer composite wrappers, cigarette butts",
        "recommendation": "Tie securely in trash bags and place in black/grey municipal bins for landfill or waste-to-energy.",
        "is_recyclable": False,
        "is_biodegradable": False,
        "color": "#495057",
        "badge_class": "bg-dark",
        "icon": "bi-trash3",
        "eco_tip": "Reduce general waste by adopting durable reusable items and saying no to excessive packaging."
    }
}

# Global cached model references
_loaded_model = None
_model_load_attempted = False
_active_engine_name = "Not Initialized"
_is_trained_ai = False

def extract_dense_features(img_pil):
    """
    Extract normalized high-dimensional visual feature vector:
    - Foreground Color Moments (Mean, Std, Skewness for R, G, B channels)
    - Relative Color Differences (Green excess, Blue excess, Warmth, Neutrality)
    - Spatial 3x3 Grid block color distributions
    - High-frequency edge gradient density & orientation
    - Specular reflectance variance
    """
    img_rgb = img_pil.convert('RGB').resize((128, 128), Image.Resampling.LANCZOS)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Background Masking: studio/white backgrounds
    is_bg = (r > 0.90) & (g > 0.90) & (b > 0.90) & (np.abs(r - g) < 0.05) & (np.abs(g - b) < 0.05)
    is_fg = ~is_bg
    
    if np.sum(is_fg) < 50:
        is_fg = np.ones_like(gray, dtype=bool)
        
    fg_r = r[is_fg]
    fg_g = g[is_fg]
    fg_b = b[is_fg]
    fg_gray = gray[is_fg]
    
    features = []
    
    # Channel Statistics on Foreground
    for ch in [fg_r, fg_g, fg_b, fg_gray]:
        features.extend([
            float(np.mean(ch)),
            float(np.std(ch)),
            float(np.percentile(ch, 10)),
            float(np.percentile(ch, 50)),
            float(np.percentile(ch, 90))
        ]) # 20 features
        
    # Relative Color Differences
    mean_r, mean_g, mean_b = float(np.mean(fg_r)), float(np.mean(fg_g)), float(np.mean(fg_b))
    features.append(mean_g - 0.5 * (mean_r + mean_b)) # Green excess (Organic)
    features.append(mean_b - 0.5 * (mean_r + mean_g)) # Blue excess (Plastic)
    features.append(mean_r - mean_b)                 # Warm/Kraft (Paper/Wood)
    features.append(1.0 - (abs(mean_r - mean_g) + abs(mean_g - mean_b) + abs(mean_r - mean_b))) # Neutrality (Metal/Glass)
    
    # Spatial 3x3 Grid Color Centers
    h_step, w_step = 128 // 3, 128 // 3
    for row in range(3):
        for col in range(3):
            sub_r = r[row*h_step:(row+1)*h_step, col*w_step:(col+1)*w_step]
            sub_g = g[row*h_step:(row+1)*h_step, col*w_step:(col+1)*w_step]
            sub_b = b[row*h_step:(row+1)*h_step, col*w_step:(col+1)*w_step]
            features.extend([float(np.mean(sub_r)), float(np.mean(sub_g)), float(np.mean(sub_b))]) # 27 features
            
    # Edge Texture Gradients
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])
    features.extend([
        float(np.mean(dx)),
        float(np.mean(dy)),
        float(np.std(dx)),
        float(np.std(dy)),
        float(np.percentile(dx, 90)),
        float(np.percentile(dy, 90))
    ]) # 6 features
    
    # Specular Reflectance
    features.append(float(np.max(gray) - np.min(gray)))
    features.append(float(np.std(gray)))
    features.append(float(np.mean(is_fg)))
    
    return np.array(features, dtype=np.float32)

def load_ml_model():
    """
    Load the trained Machine Learning Model.
    Priority 1: Deep Neural Network Pipeline (garbage_classifier_model.joblib)
    Priority 2: Keras / TensorFlow CNN Model (garbage_classifier.keras)
    Fallback: Demo Mode (Clearly marked with instructions to train)
    """
    global _loaded_model, _model_load_attempted, _active_engine_name, _is_trained_ai
    
    if _model_load_attempted:
        return _loaded_model, _active_engine_name, ORDERED_CLASSES
        
    _model_load_attempted = True
    
    # 1. Check for trained Neural Network Pipeline
    if os.path.exists(NN_MODEL_PATH):
        try:
            _loaded_model = joblib.load(NN_MODEL_PATH)
            _active_engine_name = "Deep Neural Network Vision Classifier (Trained)"
            _is_trained_ai = True
            print(f"[AI Model] Successfully loaded trained Neural Network model from {NN_MODEL_PATH}")
            return _loaded_model, _active_engine_name, ORDERED_CLASSES
        except Exception as e:
            print(f"[AI Model Warning] Failed to load joblib model: {e}")
            
    # 2. Check for Keras CNN model
    if os.path.exists(KERAS_MODEL_PATH):
        try:
            import tensorflow as tf
            _loaded_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
            _active_engine_name = "TensorFlow / Keras CNN (Trained)"
            _is_trained_ai = True
            print(f"[AI Model] Successfully loaded trained Keras CNN model from {KERAS_MODEL_PATH}")
            return _loaded_model, _active_engine_name, ORDERED_CLASSES
        except Exception as e:
            print(f"[AI Model Warning] Failed to load Keras model: {e}")
            
    # 3. Fallback: Demo Mode
    _loaded_model = None
    _active_engine_name = "Demo Mode (Untrained - Run train_model.py to train)"
    _is_trained_ai = False
    print(f"[AI Model] {_active_engine_name}")
    return None, _active_engine_name, ORDERED_CLASSES

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load image, resize to 224x224, and convert to normalized tensor.
    """
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return img, img_array

def classify_garbage_image(image_path):
    """
    Run garbage classification on input image.
    Returns structured results dictionary with top predictions and probabilities.
    """
    model, engine_name, classes_list = load_ml_model()
    img_pil, _ = preprocess_image(image_path, target_size=(224, 224))
    
    probabilities = {}
    
    if model is not None and hasattr(model, 'predict_proba'):
        # Neural Network Pipeline
        feat_vector = extract_dense_features(img_pil).reshape(1, -1)
        raw_probs = model.predict_proba(feat_vector)[0]
        for idx, prob in enumerate(raw_probs):
            cat_name = classes_list[idx] if idx < len(classes_list) else f"Class_{idx}"
            probabilities[cat_name] = float(prob)
            
    elif model is not None and hasattr(model, 'predict'):
        # Keras Model
        batch = np.expand_dims(np.array(img_pil.resize((224, 224))) / 255.0, axis=0)
        raw_preds = model.predict(batch, verbose=0)[0]
        raw_preds = raw_preds / np.sum(raw_preds)
        for idx, prob in enumerate(raw_preds):
            cat_name = classes_list[idx] if idx < len(classes_list) else f"Class_{idx}"
            probabilities[cat_name] = float(prob)
            
    else:
        # Clearly marked Demo Mode: uniform distribution with slight variance
        for idx, cat_name in enumerate(classes_list):
            probabilities[cat_name] = 1.0 / len(classes_list)
            
    # Find winning category
    best_category = max(probabilities, key=probabilities.get)
    best_confidence = round(probabilities[best_category] * 100, 1)
    
    # Sort all classes by probability descending
    formatted_probabilities = {
        cat: round(prob * 100, 1)
        for cat, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    }
    
    top_predictions = [
        {"category": cat, "confidence": prob}
        for cat, prob in formatted_probabilities.items()
    ]
    
    meta = WASTE_METADATA.get(best_category, WASTE_METADATA["General Waste"])
    
    return {
        "category": best_category,
        "confidence": best_confidence,
        "is_recyclable": meta["is_recyclable"],
        "is_biodegradable": meta["is_biodegradable"],
        "recommendation": meta["recommendation"],
        "description": meta["description"],
        "examples": meta["examples"],
        "eco_tip": meta["eco_tip"],
        "badge_class": meta["badge_class"],
        "icon": meta["icon"],
        "color": meta["color"],
        "probabilities": formatted_probabilities,
        "top_predictions": top_predictions,
        "engine": engine_name,
        "is_real_ai": _is_trained_ai
    }
