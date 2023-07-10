
"""
swat-s1 plc2
"""

from minicps.devices import PLC
from utils import PLC2_DATA, STATE, PLC2_PROTOCOL
from utils import PLC_SAMPLES, PLC_PERIOD_SEC
from utils import IP

import time

import shlex
import subprocess
from cpppo.server.enip.get_attribute import proxy_simple



PLC1_ADDR = IP['plc1']
PLC2_ADDR = IP['plc2']
PLC3_ADDR = IP['plc3']

FIT201_2 = ('FIT201', 2)


class SwatPLC2(PLC):

    def pre_loop(self, sleep=0.1):
        print 'DEBUG: swat-s1 plc2 enters pre_loop'
        print

        time.sleep(sleep)

    def main_loop(self):
        """plc2 main loop.

            - read flow level sensors #2
            - update interal enip server
        """

        print 'DEBUG: swat-s1 plc2 enters main_loop.'
        print

        # TODO: stop original server from being hosted
        # subprocess.Popen.kill()
        # try:
        #     server.kill()
        # except Exception as error:
        #     print 'ERROR stop enip server: ', error

        # Start enip server with dummy values (to be updated for each "PLC_SAMPLE" loop)
        tag_string = 'FIT201:2@22/1/1=REAL'
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
            print 'ERROR plc2 error starting server: ', error
            exit(1)

        via = proxy_simple('192.168.1.20')
        fit201_tag = '@22/1/1'
        

        # with client.connector(host=PLC2_ADDR, port=44818) as conn:
        #     # Set the value of the tag
        #     tag_name = '@22/1/1'
        #     new_value = 10
        #     conn.write([(tag_name, new_value)])
        

        count = 0
        while(count <= PLC_SAMPLES):

            fit201 = float(self.get(FIT201_2))
            print "DEBUG PLC2 - get fit201: %f" % fit201
            # self.send(FIT201_2, fit201, PLC2_ADDR) # updates value of fit201 hosted on PLC2's server (i think)
            with via: result, = via.read([(fit201_tag + '=(REAL)' + str(fit201), fit201_tag)])

            print 'this is the return value: %s' % result

            # send_status = self.send(FIT201_2, fit201, PLC1_ADDR)
            # if send_status == None or TypeError:
            #     print "\n\n\n\n\nDEBUG PLC2 - FAILED TO SEND FIT201 TO PLC1: FIT201_2 is not tuple\n\n\n\n\n"
            #     print "fit201: ", fit201
            # else:
            #     print "\n\n\n\n\nDEBUG PLC2 - send fit201: %f\n\n\n\n\n" % fit201

            # fit201 = self.receive(FIT201_2, PLC2_ADDR)
            # print "DEBUG PLC2 - receive fit201: ", fit201

            time.sleep(PLC_PERIOD_SEC)
            count += 1

        print 'DEBUG swat plc2 shutdown'


if __name__ == "__main__":

    # notice that memory init is different form disk init

    plc2 = SwatPLC2(
        name='plc2',
        state=STATE, # plc state, from sqlite database
        protocol=PLC2_PROTOCOL,
        memory=PLC2_DATA,
        disk=PLC2_DATA)
