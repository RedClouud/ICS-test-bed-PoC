from cpppo.server.enip import client
from utils import PLC2_TAGS, IP

# PLC2 details
PLC2_ADDR = IP['plc2']
PLC2_PORT = 44818
TAG_NAME = PLC2_TAGS[0][0]

# Create a client connection to PLC2
with client.connector(host=PLC2_ADDR, port=PLC2_PORT) as conn:
    # Read the value of FIT201 tag
    result = conn.read(TAG_NAME)

    # Check if the read operation was successful
    if result:
        print "result: ", result
    else:
        print "Failed to read FIT201"
