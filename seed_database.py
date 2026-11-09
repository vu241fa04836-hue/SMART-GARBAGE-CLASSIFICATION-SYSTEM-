"""
Seed database with realistic initial classification records across all 7 categories
to provide immediate visual insights in the dashboard.
"""

import database
from datetime import datetime, timedelta
import random

def seed():
    database.init_db()
    
    # Check if database already has data
    existing = database.get_all_classifications(limit=10)
    if len(existing) >= 7:
        print("[Database] Already seeded with records.")
        return
        
    sample_records = [
        ("water_bottle_plastic.jpg", "Plastic Waste", 99.2, "Empty contents, rinse clean, flatten, and deposit into designated blue plastic recycling bins.", True, False),
        ("apple_organic.jpg", "Organic Waste", 98.4, "Compost at home or place into the green organic waste bin for municipal compost & bio-fertilizer.", False, True),
        ("cardboard_paper.jpg", "Paper Waste", 96.1, "Keep clean and dry. Flatten cardboard cartons and bundle newspapers for paper pulp recycling.", True, True),
        ("soda_can_metal.jpg", "Metal Waste", 98.5, "Rinse to remove food residue and drop off in metal recycling bins or local scrap collection.", True, False),
        ("glass_bottle.jpg", "Glass Waste", 99.7, "Rinse containers and place in glass collection points. Wrap broken glass safely before disposal.", True, False),
        ("battery_ewaste.jpg", "E-Waste", 99.7, "Do not throw into household trash. Hand over to certified e-waste recyclers or retail take-back bins.", True, False),
        ("mixed_general.jpg", "General Waste", 89.3, "Tie securely in trash bags and place in black/grey municipal bins for landfill or waste-to-energy.", False, False),
        ("water_bottle_plastic.jpg", "Plastic Waste", 95.8, "Empty contents, rinse clean, flatten, and deposit into designated blue plastic recycling bins.", True, False),
        ("apple_organic.jpg", "Organic Waste", 94.7, "Compost at home or place into the green organic waste bin for municipal compost & bio-fertilizer.", False, True),
        ("cardboard_paper.jpg", "Paper Waste", 91.2, "Keep clean and dry. Flatten cardboard cartons and bundle newspapers for paper pulp recycling.", True, True),
        ("soda_can_metal.jpg", "Metal Waste", 97.4, "Rinse to remove food residue and drop off in metal recycling bins or local scrap collection.", True, False),
        ("glass_bottle.jpg", "Glass Waste", 93.6, "Rinse containers and place in glass collection points. Wrap broken glass safely before disposal.", True, False)
    ]
    
    print(f"[Database Seeder] Seeding {len(sample_records)} realistic classification records...")
    for filename, cat, conf, rec, is_rec, is_bio in sample_records:
        database.add_classification(
            filename=filename,
            category=cat,
            confidence=conf,
            disposal_recommendation=rec,
            is_recyclable=is_rec,
            is_biodegradable=is_bio,
            all_probabilities={cat: conf, "General Waste": round(100 - conf, 1)}
        )
    print("[Database Seeder] Database seeded successfully!")

if __name__ == '__main__':
    seed()
