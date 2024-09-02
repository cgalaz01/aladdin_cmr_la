import os
from typing import Tuple

from tensorboard.plugins.hparams import api as hp

# Sortable version of HParam
class HParamS(hp.HParam):
    
    def __init__(self, name, domain=None, display_name=None, description=None):
        hp.HParam.__init__(self, name, domain, display_name, description)
        
    def __lt__(self, other):
        return self.name.lower() < other.name.lower()
    
    
def get_patient_list() -> Tuple[str]:
    """
    Returns the patient list that is located in the 'data/train' directory in
    the root folder.

    Returns
    -------
    patient_list : Tuple[str]
        The sorted patient list.

    """
    return sorted(os.listdir(os.path.join('..', 'data', 'train')))
