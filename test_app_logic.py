"""
End-to-End Application Test Script
Tests Database, Model Inference, and Flask Routes.
"""

import os
import sys
import json

print("=== STARTING APPLICATION VERIFICATION ===")

# 1. Test Database
print("\n[1/3] Testing Database Module (database.py)...")
import database
database.init_db()

rec_id = database.add_classification(
    filename="test_apple.jpg",
    category="Organic Waste",
    confidence=95.2,
    disposal_recommendation="Compost at home or dispose in green organic bins.",
    is_recyclable=False,
    is_biodegradable=True,
    all_probabilities={"Organic Waste": 95.2, "General Waste": 4.8}
)
print(f" -> Successfully inserted record ID: {rec_id}")

stats = database.get_dashboard_stats()
print(f" -> Dashboard Stats Total Count: {stats['total_count']}")
print(f" -> Category counts: {stats['category_counts']}")

history = database.get_all_classifications()
print(f" -> History records count: {len(history)}")
assert len(history) > 0, "History should not be empty!"

# 2. Test AI Model Preprocessing and Classification
print("\n[2/3] Testing AI / Computer Vision Model (model.py)...")
from model import classify_garbage_image, ORDERED_CLASSES, WASTE_METADATA

test_samples = [
    ("sample_images/apple_organic.jpg", "Organic Waste"),
    ("sample_images/water_bottle_plastic.jpg", "Plastic Waste"),
    ("sample_images/cardboard_paper.jpg", "Paper Waste"),
    ("sample_images/soda_can_metal.jpg", "Metal Waste"),
    ("sample_images/glass_bottle.jpg", "Glass Waste"),
    ("sample_images/battery_ewaste.jpg", "E-Waste")
]

for sample_path, expected_cat in test_samples:
    if os.path.exists(sample_path):
        res = classify_garbage_image(sample_path)
        print(f" -> Input: {sample_path} => Predicted: {res['category']} ({res['confidence']}%) | Recyclable: {res['is_recyclable']} | Biodegradable: {res['is_biodegradable']}")
        assert res['category'] in ORDERED_CLASSES, f"Category {res['category']} not in classes"
        assert res['confidence'] > 0.0, "Confidence should be > 0"
        assert 'recommendation' in res and len(res['recommendation']) > 5, "Recommendation should be present"

# 3. Test Flask Web Routes & API
print("\n[3/3] Testing Flask Web Application Routes (app.py)...")
from app import app

client = app.test_client()

# GET /
r_home = client.get('/')
print(f" -> GET / status: {r_home.status_code}")
assert r_home.status_code == 200, "Home page failed"

# GET /classify
r_classify = client.get('/classify')
print(f" -> GET /classify status: {r_classify.status_code}")
assert r_classify.status_code == 200, "Classify page failed"

# GET /dashboard
r_dash = client.get('/dashboard')
print(f" -> GET /dashboard status: {r_dash.status_code}")
assert r_dash.status_code == 200, "Dashboard page failed"

# GET /history
r_hist = client.get('/history')
print(f" -> GET /history status: {r_hist.status_code}")
assert r_hist.status_code == 200, "History page failed"

# GET /about
r_about = client.get('/about')
print(f" -> GET /about status: {r_about.status_code}")
assert r_about.status_code == 200, "About page failed"

# GET /export-csv
r_csv = client.get('/export-csv')
print(f" -> GET /export-csv status: {r_csv.status_code}")
assert r_csv.status_code == 200, "Export CSV failed"
assert 'text/csv' in r_csv.headers['Content-Type'], "Expected text/csv content type"

# POST /api/classify with sample filename
r_api = client.post('/api/classify', 
                    data=json.dumps({'sample_filename': 'apple_organic.jpg'}),
                    content_type='application/json')
print(f" -> POST /api/classify status: {r_api.status_code}")
assert r_api.status_code == 200, "API classify failed"
api_data = json.loads(r_api.data.decode('utf-8'))
print(f" -> API Prediction Result: {api_data['category']} ({api_data['confidence']}%)")
assert api_data['success'] is True, "API should return success: True"

# POST /api/history/delete
del_id = api_data['id']
r_del = client.post(f'/api/history/delete/{del_id}')
print(f" -> POST /api/history/delete/{del_id} status: {r_del.status_code}")
assert r_del.status_code == 200, "API delete failed"

print("\n" + "=" * 55)
print(" ALL AUTOMATED APPLICATION TESTS PASSED SUCCESSFULLY! ")
print("=" * 55)
