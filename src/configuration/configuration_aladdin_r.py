from sklearn.model_selection import ParameterGrid

from tensorboard.plugins.hparams import api as hp

from configuration.utils import HParamS, get_patient_list



class HyperParameters():
    
    def __init__(self, search_type: str):
        self.HP_MODEL = HParamS('model', hp.Discrete(['aladdin_r']))
        self.HP_EPOCHS = HParamS('epochs', hp.Discrete([500]))
        self.HP_BATCH_SIZE = HParamS('batch_size', hp.Discrete([1]))
        self.HP_LEANRING_RATE = HParamS('learning_rate', hp.Discrete([0.0001]))
        self.HP_C_DILATION = HParamS('contour_dilation_radius', hp.Discrete([1]))
        self.HP_PATIENT = HParamS('patient', hp.Discrete(get_patient_list()))
        self.HP_DATA_TYPE = HParamS('data_type', hp.Discrete([False]))  # 'nn'
        self.HP_FLOW_LOSS = HParamS('flow_loss', hp.Discrete(['energy']))
        self.HP_FLOW_LAMBDA = HParamS('flow_lambda', hp.Discrete([0.1]))
        self.HP_SEED = HParamS('seed', hp.Discrete([1456]))
        
        
        self.parameter_dict = {}
        self.parameter_dict[self.HP_MODEL] = self.HP_MODEL.domain.values
        self.parameter_dict[self.HP_EPOCHS] = self.HP_EPOCHS.domain.values
        self.parameter_dict[self.HP_BATCH_SIZE] = self.HP_BATCH_SIZE.domain.values
        self.parameter_dict[self.HP_LEANRING_RATE] = self.HP_LEANRING_RATE.domain.values
        self.parameter_dict[self.HP_C_DILATION] = self.HP_C_DILATION.domain.values
        self.parameter_dict[self.HP_PATIENT] = self.HP_PATIENT.domain.values
        self.parameter_dict[self.HP_DATA_TYPE] = self.HP_DATA_TYPE.domain.values
        self.parameter_dict[self.HP_FLOW_LOSS] = self.HP_FLOW_LOSS.domain.values
        self.parameter_dict[self.HP_FLOW_LAMBDA] = self.HP_FLOW_LAMBDA.domain.values
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