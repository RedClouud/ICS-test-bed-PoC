"""
SWaT sub1 physical process

T101 has an inflow pipe and outflow pipe, both are modeled according
to the equation of continuity from the domain of hydraulics
(pressurized liquids) and a drain orefice modeled using the Bernoulli's
principle (for the trajectories).
"""


from minicps.devices import Tank

from utils import TANK_SECTION, SYSTEM_FLOWRATE_OUT_METER_CUBED
from utils import LIT_401_M, LIT_601_M, T601_INIT_LEVEL
from utils import STATE, PP_PERIOD_SEC, PP_PERIOD_HOURS, PP_SAMPLES

import time


# SPHINX_SWAT_TUTORIAL TAGS(
MV401 = ('MV401', 4)
P401 = ('P401', 4)
LIT401 = ('LIT401', 4)
LIT601 = ('LIT601', 6)
FIT401 = ('FIT401', 4)
FIT501 = ('FIT501', 5)
# SPHINX_SWAT_TUTORIAL TAGS)


# TODO: implement orefice drain with Bernoulli/Torricelli formula
class T601(Tank):

    # def pre_loop(self):

        # SPHINX_SWAT_TUTORIAL STATE INIT(
        # self.level = self.set(LIT601, 1.0)
        # SPHINX_SWAT_TUTORIAL STATE INIT)

        # test underflow
        # self.set(MV401, 0)
        # self.set(P401, 1)
        # self.level = self.set(LIT401, 0.500)

    def main_loop(self):

        count = 0
        while(count <= PP_SAMPLES):
            lit601 = self.get(LIT601)
            fit201 = self.get(FIT501)

            new_level = self.level

            # compute water volume
            water_volume = self.section * new_level

            # inflows volumes
            # TODO: put "float(lit601) < LIT_601_M['LL']" in PLC1
            inflow = float(fit201) * PP_PERIOD_HOURS # PUMP_FLOWRATE_OUT replictes outflow from raw water tank
            print "DEBUG T601 inflow: ", inflow
            water_volume += inflow

            # outflows volumes 
            # self.set(FIT501, PUMP_FLOWRATE_OUT)
            outflow = SYSTEM_FLOWRATE_OUT_METER_CUBED * PP_PERIOD_HOURS
            print "DEBUG T601 outflow: ", outflow
            water_volume -= outflow

            # compute new water_level
            new_level = water_volume / self.section

            # level cannot be negative
            if new_level <= 0.0:
                new_level = 0.0

            # update internal and state water level
            print "DEBUG FIT601 new_level: %.5f \t delta: %.5f" % (
                new_level, new_level - self.level)
            self.level = self.set(LIT601, new_level)

            # 988 sec starting from 0.500 m
            if new_level >= LIT_601_M['HH']:
                print 'DEBUG T601 above HH count: ', count
                break

            # 367 sec starting from 0.500 m
            elif new_level <= LIT_401_M['LL']:
                print 'DEBUG T601 below LL count: ', count
                break

            count += 1
            time.sleep(PP_PERIOD_SEC)


if __name__ == '__main__':

    t601 = T601(
        name='t601',
        state=STATE,
        protocol=None,
        section=TANK_SECTION,
        level=T601_INIT_LEVEL
    )
