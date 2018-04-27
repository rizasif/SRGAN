import math

import torch.nn.functional as F
from torch import nn
from torch import cat

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()

        self.growth_rate = 16
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4)
        )
        self.block2 = DenseBlock(16, self.growth_rate)
        self.block3 = TransitionBlock(16,1)
        self.block4 = DenseBlock(16, self.growth_rate)
        self.block5 = TransitionBlock(16,1)
        self.block6 = DenseBlock(16, self.growth_rate)
        # self.block7 = TransitionBlock(16,1)
        # self.block8 = DenseBlock(16, self.growth_rate)
        # self.block9 = TransitionBlock(16,1)
        # self.block10 = DenseBlock(16, self.growth_rate)

        # self.block12 = DenseBlock(16, self.growth_rate)
        # self.block13 = TransitionBlock(16,1)
        # self.block14 = DenseBlock(16, self.growth_rate)
        # self.block15 = TransitionBlock(16,1)
        # self.block16 = DenseBlock(16, self.growth_rate)
        # self.block17 = TransitionBlock(16,1)
        # self.block18 = DenseBlock(16, self.growth_rate)
        # self.block19 = TransitionBlock(16,1)
        # self.block20 = DenseBlock(16, self.growth_rate)

        self.block11 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=2, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1, padding=0, stride=1)
        )
        

    def forward(self, x):
        # print ("----forward start------")

        # print ("netG Input Size: ", x.size())

        block1 = self.block1(x)
        # print ("Block 1 Size: ", block1.size()) # Ideal torch.Size([6, 64, 22, 22])

        block2 = self.block2(block1)
        # print ("Block 2 Size: ", block2.size())

        block3 = self.block3(block2)
        # print ("Block 3 Size: ", block3.size())

        block4 = self.block4(block3)
        # print ("Block 4 Size: ", block4.size())

        block5 = self.block5(block4)
        # print ("Block 5 Size: ", block5.size())

        block6 = self.block6(block5)
        # print ("Block 6 Size: ", block6.size())

        # block7 = self.block7(block6)
        # # print ("Block 7 Size: ", block7.size())

        # block8 = self.block8(block7)
        # # # print ("Block 8 Size: ", block8.size())

        # block9 = self.block9(block8)
        # # # # print ("Block 9 Size: ", block9.size())

        # block10 = self.block10(block9)
        # print ("Block 10 Size: ", block10.size())


        #---------------
        # block12 = self.block12(block10)
        # block13 = self.block13(block12)
        # block14 = self.block14(block13)
        # block15 = self.block15(block14)
        # block16 = self.block16(block15)
        # block17 = self.block17(block16)
        # block18 = self.block18(block17)
        # block19 = self.block19(block18)
        # block20 = self.block20(block19)
        # print ("Block 20 Size: ", block20.size())
        #--------




        block11 = self.block11(block6)
        # print ("Block 11 Size: ", block11.size())

        # print ("----forward end------")
        
        # Desired Size torch.Size([6, 3, 88, 88])
        out = F.sigmoid(block11)
        # print("netG Output Size: ", out.size())

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        # Desired Size: [6, 3, 88, 88]
        # print("netD Input Tensor Size: ", x.size())
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(TransitionBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels* up_scale ** 2, stride=2,
                                 kernel_size=3, padding=0)

    def forward(self, x):
        # print("Transition Input", x.size())
        x = self.deconv(x)
        # print("Transition Output", x.size())
        return x

class DenseBlock(nn.Module):
    def __init__(self, channels, growth_rate):
        super(DenseBlock, self).__init__()

        self.relu1b = nn.PReLU()
        self.relu2b = nn.PReLU()
        self.relu3b = nn.PReLU()
        self.relu4b = nn.PReLU()
        self.relu5b = nn.PReLU()
        # self.relu6b = nn.PReLU()
        # self.relu7b = nn.PReLU()

        self.relu1a = nn.PReLU()
        self.relu2a = nn.PReLU()
        self.relu3a = nn.PReLU()
        self.relu4a = nn.PReLU()
        self.relu5a = nn.PReLU()
        # self.relu6a = nn.PReLU()
        # self.relu7a = nn.PReLU()

        self.relu1c = nn.PReLU()
        self.relu2c = nn.PReLU()
        self.relu3c = nn.PReLU()
        self.relu4c = nn.PReLU()
        self.relu5c = nn.PReLU()

        self.bn1c = nn.BatchNorm2d(channels)
        self.bn2c = nn.BatchNorm2d(channels)
        self.bn3c = nn.BatchNorm2d(channels)
        self.bn4c = nn.BatchNorm2d(channels)
        self.bn5c = nn.BatchNorm2d(channels)
        # self.bn6b = nn.BatchNorm2d(channels*6)
        # self.bn7b = nn.BatchNorm2d(channels*7)

        self.bn1a = nn.BatchNorm2d(channels*1)
        self.bn2a = nn.BatchNorm2d(channels*2)
        self.bn3a = nn.BatchNorm2d(channels*3)
        self.bn4a = nn.BatchNorm2d(channels*4)
        self.bn5a = nn.BatchNorm2d(channels*5)
        # self.bn6a = nn.BatchNorm2d(channels*6)
        # self.bn7a = nn.BatchNorm2d(channels*7)

        self.bn1b = nn.BatchNorm2d(channels*1)
        self.bn2b = nn.BatchNorm2d((channels*2)/2)
        self.bn3b = nn.BatchNorm2d((channels*3)/2)
        self.bn4b = nn.BatchNorm2d((channels*4)/2)
        self.bn5b = nn.BatchNorm2d((channels*5)/2)

        
        self.conv1c = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        self.conv2c = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        self.conv3c = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        self.conv4c = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        self.conv5c = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        # self.conv6b = nn.Conv2d(channels*6, channels, stride=1, padding=1, kernel_size=3)
        # self.conv7b = nn.Conv2d(channels*7, channels, stride=1, padding=1, kernel_size=3)

        self.conv1a = nn.Conv2d(channels*1, channels, stride=1, padding=0, kernel_size=1)
        self.conv2a = nn.Conv2d(channels*2, (channels*2)/2, stride=1, padding=0, kernel_size=1)
        self.conv3a = nn.Conv2d(channels*3, (channels*3)/2, stride=1, padding=0, kernel_size=1)
        self.conv4a = nn.Conv2d(channels*4, (channels*4)/2, stride=1, padding=0, kernel_size=1)
        self.conv5a = nn.Conv2d(channels*5, (channels*5)/2, stride=1, padding=0, kernel_size=1)
        # self.conv6a = nn.Conv2d(channels*6, channels*6, stride=1, padding=0, kernel_size=1)
        # self.conv7a = nn.Conv2d(channels*7, channels*7, stride=1, padding=0, kernel_size=1)

        self.conv1b = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        self.conv2b = nn.Conv2d((channels*2)/2, channels, stride=1, padding=1, kernel_size=3)
        self.conv3b = nn.Conv2d((channels*3)/2, channels, stride=1, padding=1, kernel_size=3)
        self.conv4b = nn.Conv2d((channels*4)/2, channels, stride=1, padding=1, kernel_size=3)
        self.conv5b = nn.Conv2d((channels*5)/2, channels, stride=1, padding=1, kernel_size=3)

        # self.batch_normb = nn.BatchNorm2d(channels)
        # self.batch_norma = nn.BatchNorm2d(channels)
        # self.convb = nn.Conv2d(channels, channels, stride=1, padding=1, kernel_size=3)
        # self.conva = nn.Conv2d(channels, channels, stride=1, padding=0, kernel_size=1)
        # self.relub = nn.PReLU()
        # self.relua = nn.PReLU()

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        
        # print("Input Size: ", x.size())

        # dense = self.batch_norma(x)
        # # print("batch_norma: ", dense.size())
        # dense = self.relua(dense)
        # # print("relua: ", dense.size())
        # dense = self.conva(dense)
        # # print("conva: ", dense.size())
        # dense = self.batch_normb(dense)
        # # print("batch_normb: ", dense.size())
        # dense = self.relub(dense)
        # # print("relub: ", dense.size())
        # dense = self.convb(dense)
        # # print("convb: ", dense.size())

        # print("First Tensor size: ", dense.size())

        # Here lies the loop that slit open
        nodes = []
        nodes.append(x)

        cocat_node = cat(tuple(nodes),1)
        # print("concat_node 1 size: ", cocat_node.size())
        dense = self.bn1a(cocat_node)
        # print("dense size bn1a: ", dense.size())
        dense = self.relu1a(dense)
        # print("dense size relu1a: ", dense.size())
        dense = self.conv1a(dense)
        # print("dense size conv1a: ", dense.size())
        dense = self.bn1b(dense)
        # print("dense size bn1b: ", dense.size())
        dense = self.relu1b(dense)
        # print("dense size relu1b: ", dense.size())
        dense = self.conv1b(dense)
        # print("dense size conv1b: ", dense.size())
        dense = self.bn1c(dense)
        dense = self.relu1c(dense)
        dense = self.conv1c(dense)
        dense = self.dropout(dense)

        nodes.append(dense)

        cocat_node = cat(tuple(nodes),1)
        # print("concat_node 2 size: ", cocat_node.size())
        dense = self.bn2a(cocat_node)
        # print("dense size bn2a: ", dense.size())
        dense = self.relu2a(dense)
        # print("dense size relu2a: ", dense.size())
        dense = self.conv2a(dense)
        # print("dense size conv2a: ", dense.size())
        dense = self.bn2b(dense)
        # print("dense size bn2b: ", dense.size())
        dense = self.relu2b(dense)
        # print("dense size relu2b: ", dense.size())
        dense = self.conv2b(dense)
        # print("dense size conv2b: ", dense.size())
        dense = self.bn2c(dense)
        dense = self.relu2c(dense)
        dense = self.conv2c(dense)
        dense = self.dropout(dense)

        nodes.append(dense)

        cocat_node = cat(tuple(nodes),1)
        # print("concat_node 3 size: ", cocat_node.size())
        dense = self.bn3a(cocat_node)
        # print("dense size bn3a: ", dense.size())
        dense = self.relu3a(dense)
        # print("dense size relu3a: ", dense.size())
        dense = self.conv3a(dense)
        # print("dense size conv3a: ", dense.size())
        dense = self.bn3b(dense)
        # print("dense size bn3b: ", dense.size())
        dense = self.relu3b(dense)
        # print("dense size relu3b: ", dense.size())
        dense = self.conv3b(dense)
        # print("dense size conv3b: ", dense.size())
        dense = self.bn3c(dense)
        dense = self.relu3c(dense)
        dense = self.conv3c(dense)
        dense = self.dropout(dense)

        nodes.append(dense)

        cocat_node = cat(tuple(nodes),1)
        # print("concat_node 4 size: ", cocat_node.size())
        dense = self.bn4a(cocat_node)
        # print("dense size bn4a: ", dense.size())
        dense = self.relu4a(dense)
        # print("dense size relu4a: ", dense.size())
        dense = self.conv4a(dense)
        # print("dense size conv4a: ", dense.size())
        dense = self.bn4b(dense)
        # print("dense size bn4b: ", dense.size())
        dense = self.relu4b(dense)
        # print("dense size relu4b: ", dense.size())
        dense = self.conv4b(dense)
        # print("dense size conv4b: ", dense.size())
        dense = self.bn4c(dense)
        dense = self.relu4c(dense)
        dense = self.conv4c(dense)
        dense = self.dropout(dense)

        nodes.append(dense)

        cocat_node = cat(tuple(nodes),1)
        dense = self.bn5a(cocat_node)
        dense = self.relu5a(dense)
        dense = self.conv5a(dense)
        dense = self.bn5b(dense)
        dense = self.relu5b(dense)
        dense = self.conv5b(dense)
        dense = self.bn5c(dense)
        dense = self.relu5c(dense)
        dense = self.conv5c(dense)
        dense = self.dropout(dense)

        # nodes.append(dense)

        # cocat_node = cat(tuple(nodes),1)
        # dense = self.bn6a(cocat_node)
        # dense = self.relu6a(dense)
        # dense = self.conv6a(dense)
        # dense = self.bn6b(dense)
        # dense = self.relu6b(dense)
        # dense = self.conv6b(dense)
        # dense = self.dropout(dense)

        # nodes.append(dense)

        # cocat_node = cat(tuple(nodes),1)
        # dense = self.bn7a(cocat_node)
        # dense = self.relu7a(dense)
        # dense = self.conv7a(dense)
        # dense = self.bn7b(dense)
        # dense = self.relu7b(dense)
        # dense = self.conv7b(dense)
        # dense = self.dropout(dense)
        
        return dense