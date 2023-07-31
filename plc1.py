# -*- coding: latin-1 -*-

"""
swat-s1 plc1.py
"""

from minicps.devices import PLC
from minicps.protocols import EnipProtocol

# Info about CPS and network
from utils import PLC1_DATA, STATE, PLC1_PROTOCOL
from utils import PLC_PERIOD_SEC, PLC_SAMPLES
from utils import IP, LIT_101_M, LIT_301_M, FIT_201_THRESH

import time

# For network comms
from cpppo.server.enip.get_attribute import proxy_simple
from cpppo import logging

PLC1_ADDR = IP['plc1']
PLC2_ADDR = IP['plc2']
PLC3_ADDR = IP['plc3']

FIT101 = ('FIT101', 1)
MV101 = ('MV101', 1)
LIT101 = ('LIT101', 1)
P101 = ('P101', 1)
# interlocks to be received from plc2 and plc3
LIT301_1 = ('LIT301', 1)  # to be sent
LIT301_3 = ('LIT301', 3)  # to be received
FIT201_1 = ('FIT201', 1)
FIT201_2 = ('FIT201', 2)
MV201_1 = ('MV201', 1)
MV201_2 = ('MV201', 2)
# SPHINX_SWAT_TUTORIAL PLC1 LOGIC)

# Request value of FIT201 from PLC2
class PLC2Parameters(proxy_simple):
    PARAMETERS = dict(proxy_simple.PARAMETERS,
                      fit201_2 = proxy_simple.parameter('@22/1/1', 'REAL', 'm^3/h'),
    )

# Request value of LIT301 from PLC3
class PLC3Parameters(proxy_simple):
    PARAMETERS = dict(proxy_simple.PARAMETERS,
                      fit201_2 = proxy_simple.parameter('@22/1/3', 'REAL', 'm'),
    )

PLC2_COMMS = PLC2Parameters(host=PLC2_ADDR)
PLC3_COMMS = PLC3Parameters(host=PLC3_ADDR)


# TODO: real value tag where to read/write flow sensor
class SwatPLC1(PLC):

    def pre_loop(self, sleep=0.1):
        print 'DEBUG: swat-s1 plc1 enters pre_loop'
        print

        time.sleep(sleep)

    def main_loop(self):
        """plc1 main loop.

            - reads sensors value
            - drives actuators according to the control strategy
            - updates its enip server
        """

        print 'DEBUG: swat-s1 plc1 enters main_loop.'
        print

        count = 0
        while(count <= PLC_SAMPLES):

            # lit101 [meters]
            lit101 = float(self.get(LIT101)) # read LIT101
            print 'DEBUG plc1 lit101: %.5f' % lit101
            # self.send(LIT101, lit101, PLC1_ADDR)

            # Compare LIT101 with well defined thresholds and take a decision then update the state
            if lit101 >= LIT_101_M['HH']:
                print "WARNING PLC1 - lit101 over HH: %.2f >= %.2f." % (
                    lit101, LIT_101_M['HH'])

            if lit101 >= LIT_101_M['H']:
                # CLOSE mv101
                print "INFO PLC1 - lit101 over H -> close mv101."
                self.set(MV101, 0)
                # self.send(MV101, 0, PLC1_ADDR)

            elif lit101 <= LIT_101_M['LL']:
                print "WARNING PLC1 - lit101 under LL: %.2f <= %.2f." % (
                    lit101, LIT_101_M['LL'])

                # CLOSE p101
                print "INFO PLC1 - close p101."
                self.set(P101, 0)
                # self.send(P101, 0, PLC1_ADDR)

            elif lit101 <= LIT_101_M['L']:
                # OPEN mv101
                print "INFO PLC1 - lit101 under L -> open mv101."
                self.set(MV101, 1)
                # self.send(MV101, 1, PLC1_ADDR)

            # Start communicating with PLC2 and PLC3...

            # TODO: use it when implement raw water tank
            # Get from PLC2
            try:
                params = PLC2_COMMS.parameter_substitution("fit201_2")
                value, = PLC2_COMMS.read(params)
                time.sleep(0.1) # wait to recive the value
            except Exception as exc: print 'fit201 read error'
                # logging.warning("Access to fit201_2 at PLC2 failed: %s", exc)
                # PLC2_COMMS.close_gateway(exc)
                # raise

            fit201 = float(value[0])

        
            print "\n\nPLC2: %f" % fit201, 

                        # read from PLC2 (constant value)

            # fit201 = self.receive(FIT201_2, PLC2_ADDR) # Ask to PLC2 FIT201’s value float(self.get(FIT201_2)) # test to see if we can access the values from the database
            # if fit201 == "":
            #     print "DEBUG PLC1 - receive fit201: None"
            #     # exit(1)
            # else:
            #     fit201 = float(fit201)
            #     print "DEBUG PLC1 - receive fit201: %f" % fit201
            #     # self.send(FIT201_1, fit201, PLC1_ADDR)

            try:
                params = PLC3_COMMS.parameter_substitution("fit201_2") # ("lit301_3")
                value, = PLC3_COMMS.read(params)
                time.sleep(0.1)
            except Exception as exc:
                logging.warning("Access to lit301_3 at PLC3 failed: %s", exc)
                PLC3_COMMS.close_gateway(exc)
                raise

            lit301 = float(value[0])

        
            print "\n\nPLC3: %f" % lit301, 

            # # read from PLC3
            # lit301 = 8 # self.receive(LIT301_3, PLC3_ADDR) # float(self.get(LIT301_3))
            # if lit301 == "":
            #     print "DEBUG PLC1 - receive lit301: None"
            #     # exit(1)
            # else:
            #     lit301 = float(lit301)
            #     print "DEBUG PLC1 - receive lit301: %f" % lit301
            #     # self.send(LIT301_1, lit301, PLC1_ADDR)

            # Compare FIT201 with well defined thresholds and take a decision then update the state
            if lit301 >= LIT_301_M['H']:
                # CLOSE p101
                self.set(P101, 0)
                # self.send(P101, 0, PLC1_ADDR)
                print "INFO PLC1 - fit201 under FIT_201_THRESH " \
                      "or over LIT_301_M['H']: -> close p101."

            # Compare FIT201 with well defined thresholds and take a decision then update the state
            if lit301 <= LIT_301_M['L']:
                # OPEN p101
                self.set(P101, 1)
                # self.send(P101, 1, PLC1_ADDR)
                print "INFO PLC1 - lit301 under LIT_301_M['L'] -> open p101."

            time.sleep(PLC_PERIOD_SEC)
            count += 1

        print 'DEBUG swat plc1 shutdown'


if __name__ == "__main__":

    # notice that memory init is different form disk init
    plc1 = SwatPLC1(
        name='plc1',
        state=STATE,
        protocol=PLC1_PROTOCOL,
        memory=PLC1_DATA,
        disk=PLC1_DATA)
