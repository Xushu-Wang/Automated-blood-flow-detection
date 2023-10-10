
if __name__ == '__main__':
    from model.networks.unet import U_Net
    import torch
    from PIL import Image
    import numpy as np
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    PATH = '/Users/andywang/Desktop/Pratt Fellow/pytorch/UltrasoundImaging/checkpoints/unet/003_net_U_Net.pth'
    image_path = "/Users/andywang/Desktop/Dataset_BUSI_with_GT/train/benign (1).png"
    net = U_Net()
    net.load_state_dict(torch.load(PATH)['network'])

    image = Image.open(image_path)
    image = image.resize((256, 256))
    image_np = np.array(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0,
            std=255
        )
    ])

    image_tensor = transform(image_np)

    pred = net(image_tensor.unsqueeze(0))

    print(pred.type)

    mask = torch.round(F.sigmoid(pred.squeeze())) * 255

    plt.imshow(image, cmap='gray')
    plt.imshow(mask.detach().numpy(), cmap='jet', alpha=0.5)
    plt.show()
