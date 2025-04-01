"""
SWaT sub1 physical process

T401 has an inflow pipe and outflow pipe, both are modeled according
to the equation of continuity from the domain of hydraulics
(pressurized liquids) and a drain orefice modeled using the Bernoulli's
principle (for the trajectories).
"""


from minicps.devices import Tank

from utils import PUMP_FLOWRATE_IN, PUMP_FLOWRATE_OUT
from utils import TANK_SECTION
from utils import LIT_401_M, T401_INIT_LEVEL
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
class T401(Tank):

    # def pre_loop(self):

        # SPHINX_SWAT_TUTORIAL STATE INIT(
        # self.set(MV401, 1)
        # self.set(P401, 0)
        # self.level = self.set(LIT401, 0.800)
        # SPHINX_SWAT_TUTORIAL STATE INIT)

        # test underflow
        # self.set(MV401, 0)
        # self.set(P401, 1)
        # self.level = self.set(LIT401, 0.500)

    def main_loop(self):

        count = 0
        while(count <= PP_SAMPLES):

            new_level = self.level

            # compute water volume
            water_volume = self.section * new_level

            # inflows volumes
            mv401 = self.get(MV401)
            if int(mv401) == 1:
                self.set(FIT401, PUMP_FLOWRATE_IN)
                inflow = PUMP_FLOWRATE_IN * PP_PERIOD_HOURS
                # print "DEBUG T401 inflow: ", inflow
                water_volume += inflow
            else:
                self.set(FIT401, 0.00)

            # outflows volumes
            p401 = self.get(P401)
            if int(p401) == 1:
                self.set(FIT501, PUMP_FLOWRATE_OUT)
                outflow = PUMP_FLOWRATE_OUT * PP_PERIOD_HOURS
                # print "DEBUG T401 outflow: ", outflow
                water_volume -= outflow
            else:
                self.set(FIT501, 0.00)

            # compute new water_level
            new_level = water_volume / self.section

            # level cannot go outside of threshholds
            if new_level <= LIT_401_M['LL']: exit
            if new_level >= LIT_401_M['HH']: exit
            # print "recorded T401 level (FIT401): " + str(self.get(LIT401))
            # print "actual T401 level: " + str(new_level)
            # print "" + time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())

            # update internal and state water level
            print "DEBUG new_level: %.5f \t delta: %.5f" % (
                new_level, new_level - self.level)
            self.level = self.set(LIT401, new_level)
            # self.level = new_level
            # self.set(LIT401, 0.7)

            # 988 sec starting from 0.500 m
            if new_level >= LIT_401_M['HH']:
                print 'DEBUG T401 above HH count: ', count
                break

            # 367 sec starting from 0.500 m
            elif new_level <= LIT_401_M['LL']:
                print 'DEBUG T401 below LL count: ', count
                break

            count += 1
            time.sleep(PP_PERIOD_SEC)


if __name__ == '__main__':

    t401 = T401(
        name='t401',
        state=STATE,
        protocol=None,
        section=TANK_SECTION,
        level=T401_INIT_LEVEL
    )
