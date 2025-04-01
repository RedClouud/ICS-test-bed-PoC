
"""
swat-s1 plc6
"""

from minicps.devices import PLC
from utils import PLC6_DATA, STATE, PLC6_PROTOCOL
from utils import PLC_SAMPLES, PLC_PERIOD_SEC
from utils import IP

import time

import shlex
import subprocess
from cpppo.server.enip.get_attribute import proxy_simple


PLC4_ADDR = IP['plc4']
PLC5_ADDR = IP['plc5']
PLC6_ADDR = IP['plc6']

LIT601_3 = ('LIT601', 6)


class SwatPLC6(PLC):

    def pre_loop(self, sleep=0.1):
        print 'DEBUG: swat-s1 plc6 enters pre_loop'
        print

        time.sleep(sleep)

    def main_loop(self):
        """plc6 main loop.

            - read UF tank level from the sensor
            - update internal enip server
        """

        # Start enip server with dummy values (to be updated for each "PLC_SAMPLE" loop)
        tag_string = 'LIT601:3@22/1/3=REAL'
        cmd = shlex.split(
            'enip_server --print' +
            ' ' + tag_string
        )
        # print 'DEBUG enip _send cmd shlex list: ', cmd

        # Start server in the background
        try:
            client = subprocess.Popen(cmd, shell=False)
            # client.wait()
            for i in range (0, 1):
                time.sleep(1)
                print 'DEBUG plc6 - waiting %d seconds for server to start' % i

        except Exception as error:
            print 'ERROR plc6 error starting server: ', error
            exit(1)

        via = proxy_simple(PLC6_ADDR)
        fit201_tag = '@22/1/3'

        count = 0
        while(count <= PLC_SAMPLES):

            lit601 = float(self.get(LIT601_3))
            print "DEBUG PLC6 - get lit601: %f" % lit601
            # self.send(LIT601_3, lit601, PLC6_ADDR)

            # with via: result, = via.read([(fit301_tag + '=(REAL)' + str(lit601), fit301_tag)])
            with via: result, = via.read([(fit201_tag + '=(REAL)' + str(lit601), fit201_tag)])
            print 'this is the return value: %s' % result

            # send_status = self.send(LIT601_3, lit601, PLC4_ADDR)
            # print "DEBUG PLC6 - send lit601: %f" % lit601

            # if send_status == None or TypeError:
            #     print "\n\n\n\n\nDEBUG PLC5 - FAILED TO SEND", LIT601_3, ":", lit601, "TO", PLC4_ADDR, ": LIT601_3 is not tuple"
            #     print "type of LIT601_3: ", type(LIT601_3)
            # else:
            #     print "\n\n\n\n\nDEBUG PLC5 - send lit601: %f\n\n\n\n\n" % lit601

            time.sleep(PLC_PERIOD_SEC)
            count += 1

        print 'DEBUG swat plc6 shutdown'


if __name__ == "__main__":

    # notice that memory init is different form disk init
    plc6 = SwatPLC6(
        name='plc6',
        state=STATE,
        protocol=PLC6_PROTOCOL,
        memory=PLC6_DATA,
        disk=PLC6_DATA)
