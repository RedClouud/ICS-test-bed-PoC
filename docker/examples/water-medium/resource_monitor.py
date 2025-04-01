# Get docker stats in json
# Parse stats (there will be a lib for this)
# Save time and stats in file
# Repeat forever

# Improvement: listen to a stream of data and save when stream item received 

import subprocess
import json
import datetime
import os

TIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'

def save_to_file(data, file):
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime(TIME_FORMAT)
    data_with_time = {'timestamp': timestamp, 'data': data}
    with open(file, 'a') as f:
        json.dump(data_with_time, f)
        f.write('\n')

def main():
    command = ['docker', 'stats', '--format', 'json']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)
    
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime(TIME_FORMAT)
    dir = 'datasets/resource_usage'
    file = f'{dir}/{timestamp}_resources.json'

    if not os.path.exists(dir):
        os.makedirs(dir)

    print ('Saving to file:', file)

    while True:
        output = process.stdout.readline().strip()
        if output == '':
            break
        try:
            json_data = json.loads(output)
            save_to_file(json_data, file)
        except json.JSONDecodeError:
            print('Error decoding JSON:', output)
            continue


if __name__ == '__main__':
    main()

