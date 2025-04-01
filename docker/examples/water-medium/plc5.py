
"""
swat-s1 plc5
"""

from minicps.devices import PLC
from utils import PLC5_DATA, STATE, PLC5_PROTOCOL
from utils import PLC_SAMPLES, PLC_PERIOD_SEC
from utils import IP

import time

import shlex
import subprocess
from cpppo.server.enip.get_attribute import proxy_simple



PLC4_ADDR = IP['plc4']
PLC5_ADDR = IP['plc5']
PLC6_ADDR = IP['plc6']

FIT501_2 = ('FIT501', 5)


class SwatPLC5(PLC):

    def pre_loop(self, sleep=0.1):
        print 'DEBUG: swat-s1 plc5 enters pre_loop'
        print

        time.sleep(sleep)

    def main_loop(self):
        """plc5 main loop.

            - read flow level sensors #2
            - update interal enip server
        """

        print 'DEBUG: swat-s1 plc5 enters main_loop.'
        print

        # TODO: stop original server from being hosted
        # subprocess.Popen.kill()
        # try:
        #     server.kill()
        # except Exception as error:
        #     print 'ERROR stop enip server: ', error

        # Start enip server with dummy values (to be updated for each "PLC_SAMPLE" loop)
        tag_string = 'FIT501:2@22/1/1=REAL'
        cmd = shlex.split(
            'enip_server --print' +
            ' ' + tag_string
        )

        # print 'DEBUG enip _send cmd shlex list: ', cmd

        # Start server in the background
        try:
            client = subprocess.Popen(cmd, shell=False)
            # client.wait()
            time.sleep(1) # wait for server to start

        except Exception as error:
            print 'ERROR plc5 error starting server: ', error
            exit(1)

        via = proxy_simple(PLC5_ADDR)
        fit501_tag = '@22/1/1'
        

        # with client.connector(host=PLC5_ADDR, port=44818) as conn:
        #     # Set the value of the tag
        #     tag_name = '@22/1/1'
        #     new_value = 10
        #     conn.write([(tag_name, new_value)])
        

        count = 0
        while(count <= PLC_SAMPLES):

            fit501 = float(self.get(FIT501_2))
            print "DEBUG PLC5 - get fit501: %f" % fit501
            # self.send(FIT501_2, fit501, PLC5_ADDR) # updates value of fit501 hosted on PLC5's server (i think)
            with via: result, = via.read([(fit501_tag + '=(REAL)' + str(fit501), fit501_tag)])

            print 'this is the return value: %s' % result

            # send_status = self.send(FIT501_2, fit501, PLC4_ADDR)
            # if send_status == None or TypeError:
            #     print "\n\n\n\n\nDEBUG PLC5 - FAILED TO SEND FIT501 TO PLC4: FIT501_2 is not tuple\n\n\n\n\n"
            #     print "fit501: ", fit501
            # else:
            #     print "\n\n\n\n\nDEBUG PLC5 - send fit501: %f\n\n\n\n\n" % fit501

            # fit501 = self.receive(FIT501_2, PLC5_ADDR)
            # print "DEBUG PLC5 - receive fit501: ", fit501

            time.sleep(PLC_PERIOD_SEC)
            count += 1

        print 'DEBUG swat plc5 shutdown'


if __name__ == "__main__":

    # notice that memory init is different form disk init

    plc5 = SwatPLC5(
        name='plc5',
        state=STATE, # plc state, from sqlite database
        protocol=PLC5_PROTOCOL,
        memory=PLC5_DATA,
        disk=PLC5_DATA)
