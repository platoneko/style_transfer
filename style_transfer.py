import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

from torchvision import models
from torchvision import transforms
from torchvision import utils

import copy


device = torch.device('cuda')

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


class Transferer(object):
    def __init__(self, content_img, style_img, imsize):
        self.loader = transforms.Compose([transforms.Resize(imsize),
                                          transforms.ToTensor()])
        self.content_img = self.image_loader(content_img)
        self.style_img = self.image_loader(style_img)
        self.cnn = copy.deepcopy(cnn)

    def image_loader(self, path):
        im = Image.open(path)
        im = self.loader(im).unsqueeze(0)
        return im.to(device, torch.float)

    def get_model(self, content_layers, style_layers):
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
        model = nn.Sequential(normalization)

        i = 0
        content_losses = []
        style_losses = []
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                layer = nn.AvgPool2d(2, stride=2)
            else:
                raise RuntimeError("Unrecognized layer:", layer.__class__.__name__)
            model.add_module(name, layer)

            if name in content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module('content_loss_{}'.format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target = model(self.style_img).detach()
                style_loss = StyleLoss(target)
                model.add_module('style_loss_{}'.format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model)-1, -1, -1):
            if isinstance(model[i], StyleLoss) or isinstance(model[i], ContentLoss):
                break

        model = model[:(i+1)]
        return model, content_losses, style_losses

    def run(self, steps_num=300, input_img='content', style_weight=1e6,
            content_layers=content_layers_default, style_layers=style_layers_default):
        if input_img == 'content':
            input_img = self.content_img.clone()
        elif input_img == 'style':
            input_img = self.style_img.clone()
        print('Building the style transfer model..')
        model, content_losses, style_losses = self.get_model(content_layers, style_layers)
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        print('Optimizing..')
        step = 0
        while step < steps_num:
            def closure():
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight

                loss = style_score + content_score
                loss.backward()

                nonlocal step
                step += 1
                if step % 50 == 0:
                    print("Iter [{}]".format(step))
                    print("Style loss: {:4f}, Content loss: {:4f}".format(style_score.item(), content_score.item()))

                return loss
            optimizer.step(closure)

        input_img.data.clamp_(0, 1)
        return input_img


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = StyleLoss.gram_mat(target).detach()

    def forward(self, input):
        G = StyleLoss.gram_mat(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    @staticmethod
    def gram_mat(input):
        a, b, c, d = input.size()
        features = input.view(a*b, c*d)
        G = torch.mm(features, features.t())
        return G.div(a*b*c*d)


if __name__ == '__main__':
    transferer = Transferer('content2.jpg', 'style.jpg', [512, 512])
    output = transferer.run()
    utils.save_image(output, 'output22.jpg')