
"""
swat-s1 plc3
"""

from minicps.devices import PLC
from utils import PLC3_DATA, STATE, PLC3_PROTOCOL
from utils import PLC_SAMPLES, PLC_PERIOD_SEC
from utils import IP

import time

import shlex
import subprocess
from cpppo.server.enip.get_attribute import proxy_simple


PLC1_ADDR = IP['plc1']
PLC2_ADDR = IP['plc2']
PLC3_ADDR = IP['plc3']

LIT301_3 = ('LIT301', 3)


class SwatPLC3(PLC):

    def pre_loop(self, sleep=0.1):
        print 'DEBUG: swat-s1 plc3 enters pre_loop'
        print

        time.sleep(sleep)

    def main_loop(self):
        """plc3 main loop.

            - read UF tank level from the sensor
            - update internal enip server
        """

        # Start enip server with dummy values (to be updated for each "PLC_SAMPLE" loop)
        tag_string = 'LIT301:3@22/1/3=REAL'
        cmd = shlex.split(
            'enip_server --print' +
            ' ' + tag_string
        )
        # print 'DEBUG enip _send cmd shlex list: ', cmd

        # Start server in the background
        try:
            client = subprocess.Popen(cmd, shell=False)
            # client.wait()

        except Exception as error:
            print 'ERROR enip _send: ', error

        via = proxy_simple('192.168.1.30')
        fit201_tag = '@22/1/3'

        count = 0
        while(count <= PLC_SAMPLES):

            lit301 = float(self.get(LIT301_3))
            print "DEBUG PLC3 - get lit301: %f" % lit301
            # self.send(LIT301_3, lit301, PLC3_ADDR)

            # with via: result, = via.read([(fit301_tag + '=(REAL)' + str(lit301), fit301_tag)])
            with via: result, = via.read([(fit201_tag + '=(REAL)' + str(lit301), fit201_tag)])
            print 'this is the return value: %s' % result

            # send_status = self.send(LIT301_3, lit301, PLC1_ADDR)
            # print "DEBUG PLC3 - send lit301: %f" % lit301

            # if send_status == None or TypeError:
            #     print "\n\n\n\n\nDEBUG PLC2 - FAILED TO SEND", LIT301_3, ":", lit301, "TO", PLC1_ADDR, ": LIT301_3 is not tuple"
            #     print "type of LIT301_3: ", type(LIT301_3)
            # else:
            #     print "\n\n\n\n\nDEBUG PLC2 - send lit301: %f\n\n\n\n\n" % lit301

            time.sleep(PLC_PERIOD_SEC)
            count += 1

        print 'DEBUG swat plc3 shutdown'


if __name__ == "__main__":

    # notice that memory init is different form disk init
    plc3 = SwatPLC3(
        name='plc3',
        state=STATE,
        protocol=PLC3_PROTOCOL,
        memory=PLC3_DATA,
        disk=PLC3_DATA)
