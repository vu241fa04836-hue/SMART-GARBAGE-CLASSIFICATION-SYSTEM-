"""
Generate high quality sample test images for one-click classification demonstrations.
"""

import os
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_images')
os.makedirs(SAMPLE_DIR, exist_ok=True)

def generate_samples():
    print("[Sample Generator] Generating one-click test sample images...")
    
    # 1. Organic: Green Apple
    img_apple = Image.new('RGB', (224, 224), color=(242, 244, 243))
    draw = ImageDraw.Draw(img_apple)
    draw.ellipse([50, 50, 170, 170], fill=(65, 185, 60))
    draw.rectangle([106, 25, 114, 52], fill=(100, 65, 30))
    draw.ellipse([114, 28, 140, 44], fill=(40, 130, 50))
    img_apple = img_apple.filter(ImageFilter.SMOOTH_MORE)
    img_apple.save(os.path.join(SAMPLE_DIR, 'apple_organic.jpg'), 'JPEG', quality=95)
    
    # 2. Plastic: Blue Mineral Water Bottle
    img_bottle = Image.new('RGB', (224, 224), color=(242, 244, 243))
    draw = ImageDraw.Draw(img_bottle)
    draw.rounded_rectangle([75, 65, 145, 185], radius=15, fill=(33, 150, 243), outline=(25, 118, 210), width=3)
    draw.rectangle([95, 40, 125, 65], fill=(100, 181, 246))
    draw.rectangle([92, 25, 128, 40], fill=(13, 71, 161)) # Dark blue cap
    draw.rectangle([75, 105, 145, 140], fill=(255, 255, 255))
    draw.line([(85, 122), (135, 122)], fill=(33, 150, 243), width=4)
    img_bottle = img_bottle.filter(ImageFilter.SMOOTH_MORE)
    img_bottle.save(os.path.join(SAMPLE_DIR, 'water_bottle_plastic.jpg'), 'JPEG', quality=95)
    
    # 3. Paper: Cardboard Shipping Box (on clean studio background)
    img_cardboard = Image.new('RGB', (224, 224), color=(242, 244, 243))
    draw = ImageDraw.Draw(img_cardboard)
    draw.rectangle([35, 35, 185, 185], fill=(210, 180, 140), outline=(139, 69, 19), width=3)
    draw.rectangle([95, 35, 125, 185], fill=(222, 184, 135), outline=(160, 82, 45), width=2)
    for line_y in range(55, 120, 10):
        draw.line([(50, line_y), (85, line_y)], fill=(80, 50, 20), width=2)
    img_cardboard.save(os.path.join(SAMPLE_DIR, 'cardboard_paper.jpg'), 'JPEG', quality=95)
    
    # 4. Metal: Aluminium Soda Can
    img_can = Image.new('RGB', (224, 224), color=(242, 244, 243))
    draw = ImageDraw.Draw(img_can)
    for x in range(70, 150):
        shade = int(140 + 70 * np.sin((x - 70) / 80.0 * np.pi))
        draw.line([(x, 50), (x, 185)], fill=(shade, shade, shade + 5), width=1)
    draw.rounded_rectangle([70, 50, 150, 185], radius=10, outline=(100, 100, 100), width=3)
    draw.ellipse([75, 40, 145, 55], fill=(210, 210, 215), outline=(120, 120, 120), width=2)
    draw.ellipse([100, 44, 120, 51], fill=(160, 160, 160))
    img_can.save(os.path.join(SAMPLE_DIR, 'soda_can_metal.jpg'), 'JPEG', quality=95)
    
    # 5. Glass: Glass Beverage Jar
    img_glass = Image.new('RGB', (224, 224), color=(242, 244, 243))
    draw = ImageDraw.Draw(img_glass)
    draw.rounded_rectangle([70, 50, 150, 180], radius=22, fill=(224, 247, 250), outline=(60, 180, 200), width=3)
    draw.rectangle([88, 30, 132, 50], fill=(180, 230, 240), outline=(50, 170, 190), width=2)
    draw.line([(85, 70), (85, 155)], fill=(255, 255, 255), width=3) # Specular highlight
    img_glass.save(os.path.join(SAMPLE_DIR, 'glass_bottle.jpg'), 'JPEG', quality=95)
    
    # 6. E-Waste: Battery Cylinder
    img_battery = Image.new('RGB', (224, 224), color=(242, 244, 243))
    draw = ImageDraw.Draw(img_battery)
    draw.rounded_rectangle([75, 60, 145, 180], radius=8, fill=(211, 47, 47), outline=(183, 28, 28), width=3)
    draw.rectangle([98, 42, 122, 60], fill=(220, 220, 220)) # Positive terminal nub
    draw.rectangle([75, 130, 145, 180], fill=(20, 20, 20)) # Base
    draw.line([(100, 85), (120, 85)], fill=(255, 255, 255), width=3)
    draw.line([(110, 75), (110, 95)], fill=(255, 255, 255), width=3)
    img_battery.save(os.path.join(SAMPLE_DIR, 'battery_ewaste.jpg'), 'JPEG', quality=95)
    
    # 7. General Waste: Mixed Debris
    img_general = Image.new('RGB', (224, 224), color=(242, 244, 243))
    draw = ImageDraw.Draw(img_general)
    for _ in range(30):
        px, py = np.random.randint(20, 180), np.random.randint(20, 180)
        draw.rectangle([px, py, px + np.random.randint(15, 35), py + np.random.randint(15, 35)], 
                       fill=(np.random.randint(80, 160), np.random.randint(80, 160), np.random.randint(80, 160)))
    img_general.save(os.path.join(SAMPLE_DIR, 'mixed_general.jpg'), 'JPEG', quality=90)
    
    print("[Sample Generator] Finished generating 7 test samples.")

if __name__ == '__main__':
    generate_samples()
