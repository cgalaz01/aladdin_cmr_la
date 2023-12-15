from sklearn.model_selection import ParameterGrid

from tensorboard.plugins.hparams import api as hp

# Sortable version of HParam
class HParamS(hp.HParam):
    
    def __init__(self, name, domain=None, display_name=None, description=None):
        hp.HParam.__init__(self, name, domain, display_name, description)
        
    def __lt__(self, other):
        return self.name.lower() < other.name.lower()


class HyperParameters():
    
    def __init__(self, search_type: str):
        # TODO: Load from file rather than hard-coded in this file
        self.HP_MODEL = HParamS('model', hp.Discrete(['base']))
        self.HP_EPOCHS = HParamS('epochs', hp.Discrete([10000]))
        self.HP_BATCH_SIZE = HParamS('batch_size', hp.Discrete([1]))
        self.HP_LEANRING_RATE = HParamS('learning_rate', hp.Discrete([0.0005]))
        self.HP_OPTIMISER = HParamS('optimiser', hp.Discrete(['adam']))
        self.HP_PATIENT = HParamS('patient', hp.Discrete(['C8', 'D1', 'E3', 'F2', 'G4',
                                                          'J5', 'J9', 'M10', 'PAT1',
                                                          'PAT3', 'PAT4', 'PAT7', 'PAT8',
                                                          'PAT9', 'PAT10', 'PAT12',
                                                          'S11', 'W12']))
        self.HP_SEED = HParamS('seed', hp.Discrete([1456]))
        
        
        self.parameter_dict = {}
        self.parameter_dict[self.HP_MODEL] = self.HP_MODEL.domain.values
        self.parameter_dict[self.HP_EPOCHS] = self.HP_EPOCHS.domain.values
        self.parameter_dict[self.HP_BATCH_SIZE] = self.HP_BATCH_SIZE.domain.values
        self.parameter_dict[self.HP_LEANRING_RATE] = self.HP_LEANRING_RATE.domain.values
        self.parameter_dict[self.HP_OPTIMISER] = self.HP_OPTIMISER.domain.values
        self.parameter_dict[self.HP_PATIENT] = self.HP_PATIENT.domain.values
        self.parameter_dict[self.HP_SEED] = self.HP_SEED.domain.values
        
        if search_type == 'grid':
            self.parameter_space = ParameterGrid(self.parameter_dict)
        else:
            raise ValueError('Invalid \'search_type\' input. Given: {}'.format(search_type))
        
        
    def __iter__(self):
        parameter_list = list(self.parameter_space)
        for parameter in parameter_list:
            yield parameter


if __name__ == '__main__':
    config = HyperParameters(search_type='grid')
    for hparams in config:
        print(hparams)