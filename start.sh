# Jack's janky start script

# Reset environment
rm swat_s1_db.sqlite
python2 init.py

docker exec trist-plc2-1 python2 plc2.py &
wait 1
docker exec trist-plc2-1 python2 plc2.py &
wait 1

docker exec trist-plc3-1 python2 plc3.py &
wait 1
docker exec trist-plc3-1 python2 plc3.py &
wait 1

docker exec trist-plc1-1 python2 plc1.py &
wait 1
docker exec trist-plc1-1 python2 plc1.py &

echo done
