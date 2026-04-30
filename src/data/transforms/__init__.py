from src.utils.decorators import ClassRegister

reg_transforms = ClassRegister('Transforms')

from .to_onehot import *
from .direction import *
from .ifh_transforms import *
from .align_labels import *