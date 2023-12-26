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
    return sorted(os.listdir(os.path.join('data', 'train')))

def get_gt_patient_list() -> Tuple[str]:
    """
    Returns the patient list that is located in the 'data/train' directory in
    the root folder and expected to have ground truth segmentation maps for all
    cardiac phases.

    Returns
    -------
    gt_patient_list : Tuple[str]
        The patient list that has ground truth segmentations.

    """
    expected_gt_patients = ['C8', 'D1', 'E3', 'PAT1', 'PAT4', 'PAT7']
    gt_patient_list = []
    # Check if they exist in the folder
    for patient in get_patient_list():
        if patient in expected_gt_patients:
            gt_patient_list.append(patient)
            
    return gt_patient_list