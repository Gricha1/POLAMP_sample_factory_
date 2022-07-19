'''
from sample_factory.algorithms.appo.model_utils import get_obs_shape, EncoderBase, ResBlock, nonlinearity
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements

from sample_factory.utils.utils import log

from torch import nn as nn

from utils.config_validation import ExperimentSettings
'''
from sample_factory.algorithms.appo.model_utils import get_obs_shape, EncoderBase, ResBlock, nonlinearity
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements

from sample_factory.utils.utils import log

from torch import nn as nn
import torch.nn.functional as F

#from utils.config_validation import ExperimentSettings
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResnetEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)
        # noinspection Pydantic
        #settings: ExperimentSettings = ExperimentSettings(**cfg.full_config['experiment_settings'])

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]
        log.debug('Num input channels: %d', input_ch)

        #resnet_conf = [[settings.pogema_encoder_num_filters, settings.pogema_encoder_num_res_blocks]]
        resnet_conf = [[64, 3]]

        self.inchannel = input_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        layers = []
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        for layer_ in [self.layer1, self.layer2, self.layer3, self.layer4]:
            layers.extend(layer_)
        layers.append(nn.AvgPool2d(4))
        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)
    
        self.init_fc_blocks(self.conv_head_out_size)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return layers

    def forward(self, x):
        #if isinstance(out, dict):
        #    out = out['obs']
        #out = self.conv1(out)
        #out = self.layer1(out)
        #out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(out)
        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        #out = self.forward_fc_blocks(out)
        #out = self.fc(out)
        #return out
        if isinstance(x, dict):
            x = x['obs']
        #print("DEBUG x:", x.shape)
        x = x[:, :2, :, :]
        x = self.conv_head(x)
        #x = F.avg_pool2d(x, 4)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.forward_fc_blocks(x)
        return x