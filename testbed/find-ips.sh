echo "Checking IPs..."

echo "Attacker:"
docker exec testbed-attacker-1 hostname -I
echo "PLC1:"
docker exec testbed-plc1-1 hostname -I
echo "PLC2:"
docker exec testbed-plc2-1 hostname -I
echo "PLC3:"
docker exec testbed-plc3-1 hostname -I

echo "Pinging from PLC1..."
docker exec testbed-plc1-1 ping -c1 testbed-plc2-1
docker exec testbed-plc1-1 ping -c1 testbed-plc3-1
docker exec testbed-plc1-1 ping -c1 testbed-attacker-1
