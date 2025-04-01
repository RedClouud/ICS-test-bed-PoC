# -*- coding: latin-1 -*-

"""
swat-s1 plc4.py
"""

from minicps.devices import PLC
from minicps.protocols import EnipProtocol

# Info about CPS and network
from utils import PLC4_DATA, STATE, PLC4_PROTOCOL
from utils import PLC_PERIOD_SEC, PLC_SAMPLES
from utils import IP, LIT_401_M, LIT_601_M

import time

# For network comms
from cpppo.server.enip.get_attribute import proxy_simple
from cpppo import logging

PLC4_ADDR = IP['plc4']
PLC5_ADDR = IP['plc5']
PLC6_ADDR = IP['plc6']

FIT401 = ('FIT401', 4)
MV401 = ('MV401', 4)
LIT401 = ('LIT401', 4)
P401 = ('P401', 4)
# interlocks to be received from plc5 and plc6
LIT601_1 = ('LIT601', 4)  # to be sent
LIT601_3 = ('LIT601', 6)  # to be received
FIT501_1 = ('FIT501', 4)
FIT501_2 = ('FIT501', 5)
# SPHINX_SWAT_TUTORIAL PLC4 LOGIC)

# Request value of FIT501 from PLC5
class PLC5Parameters(proxy_simple):
    PARAMETERS = dict(proxy_simple.PARAMETERS,
                      fit501_2 = proxy_simple.parameter('@22/1/1', 'REAL', 'm^3/h'),
    )

# Request value of LIT601 from PLC6
class PLC6Parameters(proxy_simple):
    PARAMETERS = dict(proxy_simple.PARAMETERS,
                      fit501_2 = proxy_simple.parameter('@22/1/3', 'REAL', 'm'),
    )

PLC5_COMMS = PLC5Parameters(host=PLC5_ADDR)
PLC6_COMMS = PLC6Parameters(host=PLC6_ADDR)


# TODO: real value tag where to read/write flow sensor
class SwatPLC4(PLC):

    def pre_loop(self, sleep=0.1):
        print 'DEBUG: swat-s1 plc4 enters pre_loop'
        print

        time.sleep(sleep)

    def main_loop(self):
        """plc4 main loop.

            - reads sensors value
            - drives actuators according to the control strategy
            - updates its enip server
        """

        print 'DEBUG: swat-s1 plc4 enters main_loop.'
        print

        count = 0
        while(count <= PLC_SAMPLES):

            # lit401 [meters]
            lit401 = float(self.get(LIT401)) # read LIT401
            print 'DEBUG plc4 lit401: %.5f' % lit401
            # self.send(LIT401, lit401, PLC4_ADDR)

            # Compare LIT401 with well defined thresholds and take a decision then update the state
            if lit401 >= LIT_401_M['HH']:
                print "WARNING PLC4 - lit401 over HH: %.2f >= %.2f." % (
                    lit401, LIT_401_M['HH'])

            if lit401 >= LIT_401_M['H']:
                # CLOSE mv401
                print "INFO PLC4 - lit401 over H -> close mv401."
                self.set(MV401, 0)
                # self.send(FIT401, 0, PLC4_ADDR)

            elif lit401 <= LIT_401_M['LL']:
                print "WARNING PLC4 - lit401 under LL: %.2f <= %.2f." % (
                    lit401, LIT_401_M['LL'])

                # CLOSE p401
                print "INFO PLC4 - close p401."
                self.set(P401, 0)
                # self.send(P401, 0, PLC4_ADDR)

            elif lit401 <= LIT_401_M['L']:
                # OPEN mv401
                print "INFO PLC4 - lit401 under L -> open mv401."
                self.set(MV401, 1)
                # self.send(FIT401, 1, PLC4_ADDR)

            # Start communicating with PLC5 and PLC6...

            # TODO: use it when implement raw water tank
            # Get from PLC5
            try:
                params = PLC5_COMMS.parameter_substitution("fit501_2")
                value, = PLC5_COMMS.read(params)
                time.sleep(0.1) # wait to recive the value
            except Exception as exc: print 'fit501 read error'
                # logging.warning("Access to fit501_2 at PLC5 failed: %s", exc)
                # PLC5_COMMS.close_gateway(exc)
                # raise

            fit501 = float(value[0])

        
            print "\n\nPLC5: %f" % fit501, 

                        # read from PLC5 (constant value)

            # fit501 = self.receive(FIT501_2, PLC5_ADDR) # Ask to PLC5 FIT501’s value float(self.get(FIT501_2)) # test to see if we can access the values from the database
            # if fit501 == "":
            #     print "DEBUG PLC4 - receive fit501: None"
            #     # exit(1)
            # else:
            #     fit501 = float(fit501)
            #     print "DEBUG PLC4 - receive fit501: %f" % fit501
            #     # self.send(FIT501_1, fit501, PLC4_ADDR)

            try:
                params = PLC6_COMMS.parameter_substitution("fit501_2") # ("lit601_3")
                value, = PLC6_COMMS.read(params)
                time.sleep(0.1)
            except Exception as exc:
                logging.warning("Access to lit601_3 at PLC6 failed: %s", exc)
                PLC6_COMMS.close_gateway(exc)
                raise

            lit601 = float(value[0])

        
            print "\n\nPLC6: %f" % lit601, 

            # # read from PLC6
            # lit601 = 8 # self.receive(LIT601_3, PLC6_ADDR) # float(self.get(LIT601_3))
            # if lit601 == "":
            #     print "DEBUG PLC4 - receive lit601: None"
            #     # exit(1)
            # else:
            #     lit601 = float(lit601)
            #     print "DEBUG PLC4 - receive lit601: %f" % lit601
            #     # self.send(LIT601_1, lit601, PLC4_ADDR)

            # Compare FIT501 with well defined thresholds and take a decision then update the state
            if lit601 >= LIT_601_M['H']:
                # CLOSE p401
                self.set(P401, 0)
                # self.send(P401, 0, PLC4_ADDR)
                print "INFO PLC4 - fit501 under FIT_201_THRESH " \
                      "or over LIT_601_M['H']: -> close p401."

            # Compare FIT501 with well defined thresholds and take a decision then update the state
            if lit601 <= LIT_601_M['L']:
                # OPEN p401
                self.set(P401, 1)
                # self.send(P401, 1, PLC4_ADDR)
                print "INFO PLC4 - lit601 under LIT_601_M['L'] -> open p401."

            time.sleep(PLC_PERIOD_SEC)
            count += 1

        print 'DEBUG swat plc4 shutdown'


if __name__ == "__main__":

    # notice that memory init is different form disk init
    plc4 = SwatPLC4(
        name='plc4',
        state=STATE,
        protocol=PLC4_PROTOCOL,
        memory=PLC4_DATA,
        disk=PLC4_DATA)
