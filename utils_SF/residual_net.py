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
import torch

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
        #resnet_conf = [[64, 3]]

        self.inchannel = input_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_ch, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        #layers = [self.conv1]
        #layers = []
        #self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        #self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        #self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        #self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        #self.layer2 = []
        #self.layer3 = []
        #self.layer4 = []
        #for layer_ in [self.layer1, self.layer2, self.layer3, self.layer4]:
        #    layers.extend(layer_)
        #layers.append(nn.AvgPool2d(4))

        #DEBUG
        activation = nonlinearity(self.cfg)
        conv_filters = [[input_ch, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]

        agent_layers = []
        for layer in conv_filters:
            inp_ch, out_ch, filter_size, stride = layer
            agent_layers.append(nn.Conv2d(inp_ch, out_ch, 
                                        filter_size, stride=stride))
            agent_layers.append(activation)
        self.conv_head_agent = nn.Sequential(*agent_layers)

        static_obst_layers = []
        for layer in conv_filters:
            inp_ch, out_ch, filter_size, stride = layer
            static_obst_layers.append(nn.Conv2d(inp_ch, out_ch, 
                                            filter_size, stride=stride))
            static_obst_layers.append(activation)
        self.conv_head_static_obst = nn.Sequential(*static_obst_layers)

        dynamic_obst_layers = []
        for layer in conv_filters:
            inp_ch, out_ch, filter_size, stride = layer
            dynamic_obst_layers.append(nn.Conv2d(inp_ch, out_ch, 
                                            filter_size, stride=stride))
            dynamic_obst_layers.append(activation)
        self.conv_head_dynamic_obst = nn.Sequential(*dynamic_obst_layers)

        self.agentConvOutSize = calc_num_elements(self.conv_head_agent, 
                                                        obs_shape.obs)
        self.staticObstConvOutSize = calc_num_elements(self.conv_head_static_obst, 
                                                        obs_shape.obs)
        self.dynamicObstConvOutSize = calc_num_elements(self.conv_head_dynamic_obst, 
                                                        obs_shape.obs)
        self.convLayersOutSize = self.agentConvOutSize + \
                                self.staticObstConvOutSize + \
                                self.dynamicObstConvOutSize

        #add ego car, dynamic cars extra features
        self.egoFeaturesCount = 9
        self.egoExtraOutSize = 512
        self.dynamicObstCount = 2
        self.dynamicFeaturesCount = 4
        self.dynamicExtraOutSize = 128
        
        self.ego_net = nn.Sequential(
                    nn.Linear(self.egoFeaturesCount, 
                        self.egoExtraOutSize),
                    nn.ReLU()
                    )

        self.first_dynamic_net = nn.Sequential(
                    nn.Linear(self.dynamicFeaturesCount, 
                        self.dynamicExtraOutSize),
                    nn.ReLU()
                    )
        self.second_dynamic_net = nn.Sequential(
                    nn.Linear(self.dynamicFeaturesCount, 
                        self.dynamicExtraOutSize),
                    nn.ReLU()
                    )

        self.extraFeaturesOutSize = self.egoExtraOutSize + \
                            self.dynamicObstCount * self.dynamicExtraOutSize
        
        log.debug('Convolution layers output size: %r', self.convLayersOutSize)
        log.debug('Extra layers output size: %r', self.extraFeaturesOutSize)

        #self.conv_head_out_size = 1024
        self.embedingOutSize = self.convLayersOutSize + self.extraFeaturesOutSize
        self.encoder_out_size = 1024
        self.fcForEmbeding = nn.Sequential(
                    nn.Linear(self.embedingOutSize, 
                              self.encoder_out_size),
                    nn.ReLU()
                    )
        
        log.debug('Encoder layer output size: %r', self.encoder_out_size)
        

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
        #print("DEBUG first x:", x.shape)
        #x = x[:, :1, :, :]
        #print("DEBUG second x:", x.shape)
        #x = self.conv_head(x)
        #print("DEBUG third x:", x.shape)
        #x = F.avg_pool2d(x, 4)
        #x = x.contiguous().view(-1, self.conv_head_out_size)

        static_obst_x =  \
            self.conv_head_static_obst(x[:, 0:1, :, :]).contiguous().view(-1, 
                                                    self.staticObstConvOutSize)
        dynamic_obst_x = \
            self.conv_head_dynamic_obst(x[:, 1:2, :, :]).contiguous().view(-1, 
                                                    self.dynamicObstConvOutSize)
        agent_x = self.conv_head_agent(x[:, 2:3, :, :]).contiguous().view(-1, 
                                                    self.agentConvOutSize)
        ego_x = self.ego_net(x[:, 3, 0, 0:self.egoFeaturesCount])
        #print("DEBUG ego_x shape:", ego_x.shape)
        first_dynamic_x = self.first_dynamic_net(x[:, 3, 1, 0:self.dynamicFeaturesCount])
        second_dynamic_x = self.second_dynamic_net(x[:, 3, 2, 0:self.dynamicFeaturesCount])

        x = torch.cat((static_obst_x, dynamic_obst_x, agent_x, 
                        ego_x, first_dynamic_x, second_dynamic_x), dim=-1)
        #x = torch.cat((static_obst_x, dynamic_obst_x, agent_x), dim=-1)
        x = self.fcForEmbeding(x)
 
        return x