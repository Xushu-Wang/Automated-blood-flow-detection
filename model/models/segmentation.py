import torch
import torch.nn.functional as F
from collections import OrderedDict
from model.models.base_model import BaseModel
from model.networks import get_network
from model.utils import *


class SegmentationModel(BaseModel):

    """
    Semantic Segmentation Model

    """

    def __init__(self, opts, **kwargs):
        super().__init__(opts)

        # model basic parameters
        self.pretrained = opts.pretrained
        self.inChannels = opts.input_nc
        self.outChannels = opts.output_nc
        
        self.save_dir = opts.save_dir


        # define network input and output pairs
        self.inputs = []
        self.targets = []

        self.outputs = []
        self.pred_seg = []
        self.lr_policy = opts.lr_policy

        # load/define networks
        self.net = get_network(
            opts.model_type, in_channels=self.inChannels, out_channels=self.outChannels)

        # load the model if a path is specified
        if self.pretrained:
            self.path_pre_trained_model = opts.path_pre_trained_model
            self.load_network_from_path(
                self.net, self.scaler, self.path_pre_trained_model, strict=False)

        # initialize model
        else:
            # get training criterion (Cross Entropy, etc)
            self.criterion = get_criterion(opts)

            # initialize optimizers & scheduler for training
            self.optimizer = get_optimizer(opts, self.net.parameters())
            self.scheduler = get_scheduler(self.optimizer, opts)

            print('Optimizer {} is added for the model'.format(
                self.optimizer.__class__.__name__))
            print('Scheduler {} is added for optimizer'.format(
                self.scheduler.__class__.__name__))

            # print the network details
            if kwargs.get('verbose', True):
                print('Network is initialized')
                print_network(self.net)

    def name(self):
        return 'Semantic Segmentation Model'

    # Training Functions

    def set_input(self, *inputs):
        self.inputs, self.targets = inputs

    def forward(self, split):
        if split == 'train':
            self.net.train()
            self.optimize_parameters()

        elif split == 'test':
            self.net.eval()
            with torch.no_grad():
                self.outputs = self.net(self.inputs)
                # Segmentation prediction based on [0, 1]
                self.pred_seg = torch.round(F.sigmoid(self.outputs))
                self.targets = torch.round(self.targets)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.outputs = self.net(self.inputs)
        self.loss = self.criterion(self.outputs, self.targets.float())
        self.loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics=self.loss)
        else:
            self.scheduler.step()
            
        lr = self.optimizer.param_groups[0]['lr']
        print('current learning rate = %.7f' % lr)

    def get_segmentation_stats(self, split):

        if split == 'train':
            seg_stats = self.get_current_errors()

        elif split == 'test':
            self.seg_scores, self.dice_score, self.f1_score, self.specificity, self.precision, self.recall, self.jaccard = segmentation_stats(self.pred_seg, self.targets)

            seg_stats = [
                ('Binary Accuracy', self.seg_scores.detach().item()),
                ('F1 Score', self.f1_score.detach().item()),
                ('Specificity', self.specificity.detach().item()),
                ('Precision', self.precision.detach().item()),
                ('Recall', self.recall.detach().item()),
                ('Dice Score', self.dice_score),
                ('Jaccard Index', self.jaccard.detach().item())
            ]

        return OrderedDict(seg_stats)

    def get_current_errors(self):
        return OrderedDict([('Segmentation Loss', self.loss.detach().item())])

    # Utility Functions

    def save(self, epoch_label):
        self.save_network(self.net, self.net.__class__.__name__, epoch_label)










if __name__ == '__main__':
    

    def generate_random_data(batch_size, in_channels, out_channels, height, width):
        inputs = torch.rand(batch_size, in_channels, height, width)
        targets = torch.randint(
            0, out_channels + 1, (batch_size, out_channels, height, width))
        return inputs, targets

    class Options:
        def __init__(self):
            self.pretrained = False
            self.input_nc = 3
            self.output_nc = 1  # Assuming 21 classes for segmentation
            self.model_type = 'unet'  # Assuming 'resnet' is one of the model types
            self.lr_policy = 'step'  # Assuming 'step' is the learning rate policy
            # Replace with an actual path if pretrained is True
            self.path_pre_trained_model = 'path_to_pretrained_model'
            self.criterion = 'cross_entropy'
            self.lr_rate = 1e-4
            self.optim = 'adam'
            self.lr_decay_iters = 250
            self.save_dir = ""

    opts = Options()
    model = SegmentationModel(opts)

    # Set the model to evaluation mode
    model.net.eval()

    # Generate random test data
    batch_size = 4
    height, width = 256, 256
    inputs, targets = generate_random_data(
        batch_size, opts.input_nc, opts.output_nc, height, width)
    
    print(inputs)
    print(targets)

    # Perform forward pass on the model with test data
    with torch.no_grad():
        model.set_input(inputs, targets)
        model.forward('test')

    
    # Perform some assertions or checks on the output
    assert model.outputs.shape == (batch_size, opts.output_nc, height, width)
    assert model.pred_seg.shape == (batch_size, opts.output_nc, height, width)
    assert model.targets.shape == (batch_size, opts.output_nc, height, width)
    print("######################")
    print(model.outputs)
    print((model.pred_seg > 0).sum())



    # Check if segmentation stats are calculated correctly
    stats = model.get_segmentation_stats('test')
    assert 'Binary Accuracy' in stats
    assert 'Dice Score' in stats
    assert 'Jaccard Index' in stats

    
    print(stats)


    print("SegmentationModel test passed!")
    

