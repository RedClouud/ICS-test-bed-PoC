# Takes a reading from the state database every second and stores it in a csv file

import time
import csv
import sqlite3
import os

values = {
    'FIT101': 0,
    'MV101': 0,
    'LIT101': 0,
    'P101': 0,
    'FIT201': 0,
    'LIT301': 0
}

# Connect to the database
conn = sqlite3.connect('swat_s1_db.sqlite')
c = conn.cursor()
print "Connected to database"

current_time = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())
dataset_tag = "normal"
dataset_path = "./datasets/%s_%s.csv" % (current_time, dataset_tag)

# Create the csv file
if not os.path.exists("./datasets"):
    os.mkdir("./datasets")

if os.path.exists(dataset_path):
    os.remove(dataset_path)

with open(dataset_path, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['time', 'FIT101', 'MV101', 'LIT101', 'P101', 'FIT201', 'LIT301'])
print "Created csv file"

print "Starting data collection..."

# Take a reading every second
while True:
    # Get the current time
    current_time = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())

    # Get the current state
    c.execute('SELECT * FROM swat_s1')
    state = c.fetchall()

    for component in state:
        name = component[0]
        value = component[2]
        values[name] = value

    print(current_time, values)

    # Write the state to the csv file
    with open(dataset_path, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([current_time, values['FIT101'], values['MV101'], values['LIT101'], values['P101'], values['FIT201'], values['LIT301']])
   
    # Wait a second
    time.sleep(1)


