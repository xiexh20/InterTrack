from .smplpytorch import SMPL_Layer
from .smpl_generator import SMPLHGenerator
from .const import SMPL_MODEL_ROOT

def get_smpl(gender, hands):
    "simple wrapper to get SMPL model"
    return SMPL_Layer(model_root=SMPL_MODEL_ROOT,
               gender=gender, hands=hands)