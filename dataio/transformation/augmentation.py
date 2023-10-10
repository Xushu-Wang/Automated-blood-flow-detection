import albumentations as A
from pprint import pprint



class Augmentation:
    
    """
    Augmentation Class
    
    Augmentation including randomflip, shifting, rotation, scaling, blur, sharpen, and motion blur
    
    """

    def __init__(self, name):
        self.name = name
        self.shift_val = (0.1, 0.1)
        self.random_flip_prob = 0.5
        self.rotate_val = 15
        self.scale_val = (0.8, 1.2)
        self.gaussian_noise = (10.0, 50.0)

    def initialize(self, opts):
        aug_opts = getattr(opts, self.name)

        if hasattr(aug_opts, 'shift_val'):
            self.shift_val = aug_opts.shift
        if hasattr(aug_opts, 'random_flip_prob'):
            self.random_flip_prob = aug_opts.random_flip_prob
        if hasattr(aug_opts, 'rotate_val'):
            self.rotate_val = aug_opts.rotate
        if hasattr(aug_opts, 'scale_val'):
            self.scale_val = aug_opts.scale
        if hasattr(aug_opts, 'gaussian'):
            self.gaussian_noise = aug_opts.gaussian

    def ultrasound_transform(self):

        train_transform = [

            A.HorizontalFlip(p=self.random_flip_prob),

            A.ShiftScaleRotate(
                scale_limit=self.scale_val[1],
                rotate_limit=self.rotate_val,
                shift_limit_x=self.shift_val[0],
                shift_limit_y=self.shift_val[1],
                p=1
            )

            # A.transforms.GaussNoise(p=0.2),


            # A.OneOf(
            #     [
            #         A.transforms.Sharpen(p=1),
            #         A.Blur(blur_limit=3, p=1),
            #         A.MotionBlur(blur_limit=3, p=1),
            #     ],
            #     p=0.9,
            # )
        ]

        train_transform = A.Compose(train_transform)

        test_transform = [
            A.transforms.GaussNoise(p=1, var_limit=self.gaussian_noise) # Gaussian Noise for imitating low mechanical index
        ]

        test_transform = A.Compose(test_transform)

        return {'train': train_transform, 'test': test_transform}

    def phantom_transform(self):

        pass

    def get_augmentation(self):

        return {
            'BUSI': self.ultrasound_transform,
            'Phantom': self.phantom_transform
        }[self.name]()

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')



if __name__ == '__main__':
    
    """Test Case for augmentation"""
    
    import numpy as np
    
    random_image = np.random.rand(64, 64, 3)

    # Create an instance of the Augmentation class with 'BUSI' name
    augmentation = Augmentation('BUSI')


    # Get the augmentation pipeline for the 'BUSI' dataset
    augmentation_pipeline = augmentation.get_augmentation()['train']
    train_image = augmentation_pipeline(image=random_image)['image']

    augmentation_pipeline = augmentation.get_augmentation()['test']
    test_image = augmentation_pipeline(image=random_image)['image']

    # Print the original and augmented images for comparison
    print("Original Image:", random_image)
    print("Augmented Train Image:", train_image)
    print("Augmented Test Image:", test_image)