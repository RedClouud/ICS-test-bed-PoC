"""
SWaT sub1 physical process

T101 has an inflow pipe and outflow pipe, both are modeled according
to the equation of continuity from the domain of hydraulics
(pressurized liquids) and a drain orefice modeled using the Bernoulli's
principle (for the trajectories).
"""


from minicps.devices import Tank

from utils import PUMP_FLOWRATE_IN, PUMP_FLOWRATE_OUT
from utils import TANK_SECTION
from utils import LIT_101_M, T101_INIT_LEVEL
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
class T101(Tank):

    # def pre_loop(self):

        # SPHINX_SWAT_TUTORIAL STATE INIT(
        # self.set(MV101, 1)
        # self.set(P101, 0)
        # self.level = self.set(LIT101, 0.800)
        # SPHINX_SWAT_TUTORIAL STATE INIT)

        # test underflow
        # self.set(MV101, 0)
        # self.set(P101, 1)
        # self.level = self.set(LIT101, 0.500)

    def main_loop(self):

        count = 0
        while(count <= PP_SAMPLES):

            new_level = self.level

            # compute water volume
            water_volume = self.section * new_level

            # inflows volumes
            mv101 = self.get(MV101)
            if int(mv101) == 1:
                self.set(FIT101, PUMP_FLOWRATE_IN)
                inflow = PUMP_FLOWRATE_IN * PP_PERIOD_HOURS
                # print "DEBUG T101 inflow: ", inflow
                water_volume += inflow
            else:
                self.set(FIT101, 0.00)

            # outflows volumes
            p101 = self.get(P101)
            if int(p101) == 1:
                self.set(FIT201, PUMP_FLOWRATE_OUT)
                outflow = PUMP_FLOWRATE_OUT * PP_PERIOD_HOURS
                # print "DEBUG T101 outflow: ", outflow
                water_volume -= outflow
            else:
                self.set(FIT201, 0.00)

            # compute new water_level
            new_level = water_volume / self.section

            # level cannot go outside of threshholds
            if new_level <= LIT_101_M['LL']: exit
            if new_level >= LIT_101_M['HH']: exit
            # print "recorded T101 level (FIT101): " + str(self.get(LIT101))
            # print "actual T101 level: " + str(new_level)
            # print "" + time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())

            # update internal and state water level
            print "DEBUG new_level: %.5f \t delta: %.5f" % (
                new_level, new_level - self.level)
            self.level = self.set(LIT101, new_level)
            # self.level = new_level
            # self.set(LIT101, 0.7)

            # 988 sec starting from 0.500 m
            if new_level >= LIT_101_M['HH']:
                print 'DEBUG T101 above HH count: ', count
                break

            # 367 sec starting from 0.500 m
            elif new_level <= LIT_101_M['LL']:
                print 'DEBUG T101 below LL count: ', count
                break

            count += 1
            time.sleep(PP_PERIOD_SEC)


if __name__ == '__main__':

    t101 = T101(
        name='t101',
        state=STATE,
        protocol=None,
        section=TANK_SECTION,
        level=T101_INIT_LEVEL
    )
