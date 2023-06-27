# from cpppo.server.enip import client

# IP and port of the server
server_ip = '192.168.1.20'
# server_port = 44818 (this is default anyway)

from cpppo.server.enip.get_attribute import proxy_simple
from cpppo import logging

class some_sensor( proxy_simple ):
    '''A simple (non-routing) CIP device with one parameter with a
       shortcut name: 'A Sensor Parameter' '''
    PARAMETERS     = dict( proxy_simple.PARAMETERS,
        a_sensor_parameter = proxy_simple.parameter( '@22/1/1', 'INT', 'None' ),
    )

via                = some_sensor( host=server_ip )
try:
    params         = via.parameter_substitution( "A Sensor Parameter" )
    value,         = via.read( params )
except Exception as exc:
    logging.warning( "Access to remote CIP device failed: %s", exc )
    via.close_gateway( exc )
    raise
    
print( "A Sensor Parameter: %s" % ( value, ) )
