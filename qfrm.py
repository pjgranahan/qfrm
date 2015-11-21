from OptionValuation import *

from European import *
from American import *

from Asian import *
from Barrier import *
from Bermudan import *
from Binary import *
from Chooser import *
from European import *
from ForwardStart import *
from Gap import *
from Lookback import *
from LowExercisePrice import *
from PerpetualAmerican import *
from Quanto import *
from Shout import *
from VarianceSwap import *


# s = Stock(S0=30, vol=.3)
# o = European(ref=s, right='call', K=30, T=1., rf_r=.08, desc='Example from Internet')
# o.calc_px(method='BS')
# print(o.px_spec)