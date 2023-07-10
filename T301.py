"""
SWaT sub1 physical process

T101 has an inflow pipe and outflow pipe, both are modeled according
to the equation of continuity from the domain of hydraulics
(pressurized liquids) and a drain orefice modeled using the Bernoulli's
principle (for the trajectories).
"""


from minicps.devices import Tank

from utils import TANK_SECTION
from utils import LIT_101_M, LIT_301_M, T301_INIT_LEVEL
from utils import STATE, PP_PERIOD_SEC, PP_PERIOD_HOURS, PP_SAMPLES

import time


# SPHINX_SWAT_TUTORIAL TAGS(
MV101 = ('MV101', 1)
P101 = ('P101', 1)
LIT101 = ('LIT101', 1)
LIT301 = ('LIT301', 3)
FIT101 = ('FIT101', 1)
FIT201 = ('FIT201', 2)
# SPHINX_SWAT_TUTORIAL TAGS)


# TODO: implement orefice drain with Bernoulli/Torricelli formula
class T301(Tank):

    def pre_loop(self):

        # SPHINX_SWAT_TUTORIAL STATE INIT(
        self.level = self.set(LIT301, 1.0)
        # SPHINX_SWAT_TUTORIAL STATE INIT)

        # test underflow
        # self.set(MV101, 0)
        # self.set(P101, 1)
        # self.level = self.set(LIT101, 0.500)

    def main_loop(self):

        count = 0
        while(count <= PP_SAMPLES):
            lit301 = self.get(LIT301)
            fit201 = self.get(FIT201)

            new_level = self.level

            # compute water volume
            water_volume = self.section * new_level

            # inflows volumes
            # TODO: put "float(lit301) < LIT_301_M['LL']" in PLC1
            inflow = float(fit201) * PP_PERIOD_HOURS # PUMP_FLOWRATE_OUT replictes outflow from raw water tank
            print "DEBUG T301 inflow: ", inflow
            water_volume += inflow

            # outflows volumes (constant rate at 1.5m^1/h)
            # self.set(FIT201, PUMP_FLOWRATE_OUT)
            outflow = 1.5 * PP_PERIOD_HOURS
            print "DEBUG T301 outflow: ", outflow
            water_volume -= outflow

            # compute new water_level
            new_level = water_volume / self.section

            # level cannot be negative
            if new_level <= 0.0:
                new_level = 0.0

            # update internal and state water level
            print "DEBUG FIT301 new_level: %.5f \t delta: %.5f" % (
                new_level, new_level - self.level)
            self.level = self.set(LIT301, new_level)

            # 988 sec starting from 0.500 m
            if new_level >= LIT_301_M['HH']:
                print 'DEBUG T301 above HH count: ', count
                break

            # 367 sec starting from 0.500 m
            elif new_level <= LIT_101_M['LL']:
                print 'DEBUG T301 below LL count: ', count
                break

            count += 1
            time.sleep(PP_PERIOD_SEC)


if __name__ == '__main__':

    t301 = T301(
        name='t301',
        state=STATE,
        protocol=None,
        section=TANK_SECTION,
        level=T301_INIT_LEVEL
    )
