import os

class ModelOpts:
    
    """
    Model Options wrapper: Created for initializing model
    """
    
    def __init__(self):
        self.gpu_ids = [0]
        self.pretrained = False
        self.save_dir = './checkpoints/default'
        self.model_type = 'unet'
        self.input_nc = 1
        self.output_nc = 1
        self.lr_policy = None
        self.lr_rate = 1e-12
        self.lr_decay_iters = 5
        self.l2_reg_weight = 0.0
        self.feature_scale = 4
        self.path_pre_trained_model = None
        self.criterion = 'cross_entropy'
        self.type = 'seg'
        self.optim = 'adam'
  

    def initialize(self, json_opts):
        opts = json_opts

        self.pretrained = opts.pretrained
        self.save_dir = os.path.join(opts.checkpoints_dir, opts.experiment_name)
        self.model_type = opts.model_type
        self.input_nc = opts.input_nc

        if hasattr(opts, 'type'):
            self.type = opts.type
        if hasattr(opts, 'l2_reg_weight'):
            self.l2_reg_weight = opts.l2_reg_weight
        if hasattr(opts, 'lr_rate'):
            self.lr_rate = opts.lr_rate
        if hasattr(opts, 'lr_policy'):
            self.lr_policy = opts.lr_policy
        if hasattr(opts, 'lr_policy'):
            self.lr_decay_iters = opts.lr_decay_iters
        if hasattr(opts, 'feature_scale'):
            self.feature_scale = opts.feature_scale
        if hasattr(opts, 'pretrained_path'):
            self.path_pre_trained_model = opts.pretrained_path
        if hasattr(opts, 'criterion'):
            self.criterion = opts.criterion
        if hasattr(opts, "optim"):
            self.optim = opts.optim
        

    


def get_model(json_opts):
    model = None
    model_opts = ModelOpts()
    model_opts.initialize(json_opts)
    
    print('\nInitializing model {}'.format(model_opts.model_type))
    
    if model_opts.type == 'seg':
        from .segmentation import SegmentationModel
        model = SegmentationModel(model_opts)
            
    print("Model [%s] is created" % (model.name()))
    return model

    