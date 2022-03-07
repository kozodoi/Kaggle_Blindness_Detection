from efficientnet_pytorch import EfficientNet
import torch


def init_model(train      = True, 
               trn_layers = 2,
               model_name):
    
    '''
    Initialize the model
    '''
    
    ### training mode
    if train == True:
        
        # load pre-trained model
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 5)
        model.load_state_dict(torch.load('../models/model_{}.bin'.format(model_name, 1)))   
        
        # freeze first layers
        for child in list(model.children())[:-trn_layers]:
            for param in child.parameters():
                param.requires_grad = False
        
        
    ### inference mode
    if train == False:
        
        # load pre-trained model
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 5)
        model.load_state_dict(torch.load('../models/model_{}.bin'.format(model_name, 1)))   

        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
            
            
    ### return model
    return model