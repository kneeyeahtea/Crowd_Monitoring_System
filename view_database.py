import sqlite3

# Connect to database
conn = sqlite3.connect('crowd_monitoring.db')
cursor = conn.cursor()

# Get all unique locations
cursor.execute("SELECT DISTINCT location FROM crowd_data ORDER BY location")
locations = cursor.fetchall()

print(f"\n{'='*100}")
print(f"CROWD MONITORING DATABASE")
print(f"{'='*100}\n")

if len(locations) > 0:
    # Display data for each location separately
    for (location,) in locations:
        # Get records for this location
        cursor.execute("""
            SELECT * FROM crowd_data 
            WHERE location = ?
            ORDER BY timestamp DESC
        """, (location,))
        rows = cursor.fetchall()
        
        print(f"\n{'='*100}")
        print(f"📍 LOCATION: {location.upper()} - Total Records: {len(rows)}")
        print(f"{'='*100}\n")
        
        # Print column headers
        print(f"{'ID':<5} {'Timestamp':<20} {'Count':<8} {'Status':<10} {'Hotspots':<10}")
        print("-" * 100)
        
        # Print each record for this location
        for row in rows:
            print(f"{row[0]:<5} {row[1]:<20} {row[3]:<8} {row[4]:<10} {row[7]:<10}")
        
        # Get statistics for this location
        cursor.execute("""
            SELECT 
                AVG(crowd_count) as avg_count,
                MAX(crowd_count) as max_count,
                MIN(crowd_count) as min_count,
                COUNT(*) as total_records
            FROM crowd_data
            WHERE location = ?
        """, (location,))
        
        stats = cursor.fetchone()
        
        print("\n" + "-" * 100)
        print("STATISTICS:")
        print(f"  Average Count: {int(stats[0])} people")
        print(f"  Peak Count:    {int(stats[1])} people")
        print(f"  Min Count:     {int(stats[2])} people")
        print(f"  Total Records: {stats[3]}")
        print("=" * 100)
        
else:
    print("No data found! Run main.py first to process a video.")

# Overall summary
cursor.execute("SELECT COUNT(*) FROM crowd_data")
total_records = cursor.fetchone()[0]

print(f"\n{'='*100}")
print(f"OVERALL SUMMARY")
print(f"{'='*100}")
print(f"Total Locations: {len(locations)}")
print(f"Total Records:   {total_records}")
print(f"{'='*100}\n")

conn.close()
print("✅ Database view complete!\n")
