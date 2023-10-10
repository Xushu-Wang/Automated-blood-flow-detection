import os
import torch



class BaseModel():
    
    """
    Base Model
    
    """
    
    def __init__(self, opt, **kwargs):
        
        self.net = None
        self.input = None
        self.label = None
        
        self.scheduler = None
        self.optimizer = None
        
        self.pre_trained = None
        self.use_cuda = False
        
        self.save_dir = opt.save_dir
    
    def name(self):
        return 'Base Model'
    

    
    # Initialization
    def set_input(self, input):
        self.input = input
    
    def set_scheduler(self, train_opts):
        pass
    
    # Training Process
    def forward(self):
        pass
    
    def optimize(self):
        pass
    
    def update_learning_rate(self):
        pass
    
    def get_current_errors(self):
        pass
    

    # Utility Function
    
    def save_network(self, network, network_label, epoch_label):
        print('Saving the model {} at the end of epoch {}'.format(network_label, epoch_label))
        
        save_filename = '{0:03d}_net_{1}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        
        torch.save({"network": network.cpu().state_dict()}, save_path)


    def load_network(self, network, network_label, epoch_label):
        print('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
        save_filename = '{0:03d}_net_{1}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def load_network_from_path(self, network, network_filepath, strict):
        network_label = os.path.basename(network_filepath)
        epoch_label = network_label.split('_')[0]
        print('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
        
        try:
            network.load_state_dict(torch.load(network_filepath)["network"], strict=strict)
        except:
            network.load_state_dict(torch.load(network_filepath), strict=strict)
    
    def deconstructor(self):
        del self.net
        del self.input
    
    
