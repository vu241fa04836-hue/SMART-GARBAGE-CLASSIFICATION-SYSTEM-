"""
Database Management Module for Smart Garbage Classification System
Handles SQLite database operations for storing and querying classification records.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'garbage_classifier.db')

def get_connection():
    """Establish and return a SQLite connection with row dictionary access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            disposal_recommendation TEXT NOT NULL,
            is_recyclable INTEGER NOT NULL DEFAULT 0,
            is_biodegradable INTEGER NOT NULL DEFAULT 0,
            all_probabilities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_classification(filename, category, confidence, disposal_recommendation, 
                       is_recyclable=False, is_biodegradable=False, all_probabilities=None):
    """
    Insert a new classification entry into the database.
    Returns the newly created row ID.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    prob_json = json.dumps(all_probabilities) if all_probabilities is not None else None
    
    cursor.execute('''
        INSERT INTO classifications 
        (filename, category, confidence, disposal_recommendation, is_recyclable, is_biodegradable, all_probabilities, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
    ''', (
        filename,
        category,
        float(confidence),
        disposal_recommendation,
        1 if is_recyclable else 0,
        1 if is_biodegradable else 0,
        prob_json
    ))
    
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id

def get_all_classifications(limit=200, category_filter=None, search_term=None):
    """
    Retrieve classification records with optional category filtering and search.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = 'SELECT * FROM classifications WHERE 1=1'
    params = []
    
    if category_filter and category_filter != 'All':
        query += ' AND LOWER(category) = LOWER(?)'
        params.append(category_filter)
        
    if search_term:
        query += ' AND (LOWER(filename) LIKE LOWER(?) OR LOWER(disposal_recommendation) LIKE LOWER(?))'
        params.extend([f'%{search_term}%', f'%{search_term}%'])
        
    query += ' ORDER BY id DESC LIMIT ?'
    params.append(limit)
    
    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Parse json probabilities if present
    for row in rows:
        if row.get('all_probabilities'):
            try:
                row['all_probabilities'] = json.loads(row['all_probabilities'])
            except Exception:
                row['all_probabilities'] = {}
        else:
            row['all_probabilities'] = {}
            
    return rows

def get_classification_by_id(record_id):
    """Retrieve a single classification by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM classifications WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        data = dict(row)
        if data.get('all_probabilities'):
            try:
                data['all_probabilities'] = json.loads(data['all_probabilities'])
            except Exception:
                data['all_probabilities'] = {}
        return data
    return None

def get_dashboard_stats():
    """
    Calculate and return comprehensive statistics for the analytics dashboard.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total count
    cursor.execute('SELECT COUNT(*) FROM classifications')
    total_count = cursor.fetchone()[0]
    
    # Standard 7 waste categories
    standard_categories = [
        'Organic Waste',
        'Plastic Waste',
        'Paper Waste',
        'Metal Waste',
        'Glass Waste',
        'E-Waste',
        'General Waste'
    ]
    
    category_counts = {cat: 0 for cat in standard_categories}
    
    cursor.execute('''
        SELECT category, COUNT(*) as count 
        FROM classifications 
        GROUP BY category
    ''')
    for row in cursor.fetchall():
        cat = row['category']
        count = row['count']
        if cat in category_counts:
            category_counts[cat] = count
        else:
            # Match loosely if needed
            matched = False
            for sc in standard_categories:
                if sc.lower().startswith(cat.lower()) or cat.lower().startswith(sc.lower()):
                    category_counts[sc] += count
                    matched = True
                    break
            if not matched:
                category_counts['General Waste'] += count
                
    # Recyclable vs Biodegradable vs Non-recyclable
    cursor.execute('SELECT SUM(is_recyclable), SUM(is_biodegradable), AVG(confidence) FROM classifications')
    stats_row = cursor.fetchone()
    total_recyclable = stats_row[0] or 0
    total_biodegradable = stats_row[1] or 0
    avg_confidence = round(stats_row[2] or 0.0, 1)
    
    # Recent 7 classifications for preview
    cursor.execute('SELECT * FROM classifications ORDER BY id DESC LIMIT 7')
    recent_records = [dict(r) for r in cursor.fetchall()]
    
    # Recent 7 days trend
    cursor.execute('''
        SELECT DATE(created_at) as log_date, COUNT(*) as daily_count
        FROM classifications
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at) DESC
        LIMIT 7
    ''')
    daily_trends = [dict(r) for r in cursor.fetchall()]
    daily_trends.reverse() # Chronological order
    
    conn.close()
    
    return {
        'total_count': total_count,
        'category_counts': category_counts,
        'total_recyclable': total_recyclable,
        'total_biodegradable': total_biodegradable,
        'avg_confidence': avg_confidence,
        'recent_records': recent_records,
        'daily_trends': daily_trends
    }

def delete_classification(record_id):
    """Delete a classification record and return filename if existing."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT filename FROM classifications WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    filename = row['filename'] if row else None
    
    cursor.execute('DELETE FROM classifications WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    return filename

def clear_all_history():
    """Clear all records from classifications table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT filename FROM classifications')
    rows = cursor.fetchall()
    filenames = [r['filename'] for r in rows if r['filename']]
    cursor.execute('DELETE FROM classifications')
    cursor.execute('DELETE FROM sqlite_sequence WHERE name="classifications"')
    conn.commit()
    conn.close()
    return filenames

# Initialize DB on module load if needed
init_db()
