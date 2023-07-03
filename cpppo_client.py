# from cpppo.server.enip import client

# IP and port of the server
server_ip = ['192.168.1.20', '192.168.1.30']
# server_port = 44818 (this is default anyway)

from cpppo.server.enip.get_attribute import proxy_simple
from cpppo import logging

class some_sensor2( proxy_simple ):
    '''A simple (non-routing) CIP device with one parameter with a
       shortcut name: 'A Sensor Parameter' '''
    PARAMETERS     = dict( proxy_simple.PARAMETERS,
        a_sensor_parameter = proxy_simple.parameter( '@22/1/3', 'REAL', 'None' ),
    )

# class some_sensor3( proxy_simple ):
#     '''A simple (non-routing) CIP device with one parameter with a
#        shortcut name: 'A Sensor Parameter' '''
#     PARAMETERS     = dict( proxy_simple.PARAMETERS,
#         a_sensor_parameter = proxy_simple.parameter( '@22/1/2', 'INT', 'None' ),
#     )

via2                = some_sensor2( host=server_ip[0] )
# via3                = some_sensor3( host=server_ip[1] )

while (True):
    # Get from PLC2
    try:
        params         = via2.parameter_substitution( "A Sensor Parameter" )
        value,         = via2.read( params )
    except Exception as exc:
        logging.warning( "Access to remote CIP device failed: %s", exc )
        via2.close_gateway( exc )
        raise
        
    print( "\n\nPLC2: %s" % ( value, ) )

    # # Get from PLC3
    # try:
    #     params         = via3.parameter_substitution( "A Sensor Parameter" )
    #     value,         = via3.read( params )
    # except Exception as exc:
    #     logging.warning( "Access to remote CIP device failed: %s", exc )
    #     via3.close_gateway( exc )
    #     raise
        
    # print( "\n\nPLC3: %s" % ( value, ) )
