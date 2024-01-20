#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# March 2020                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# rcnn.py                                    oN88888UU[[[/;::-.        dP^
# define recurrent networks                 dNMMNN888UU[[[/;:--.   .o@P^
#                                          ,MMMMMMN888UU[[/;::-. o@^
#                                          NNMMMNN888UU[[[/~.o@P^
# Markus Ernst                             888888888UU[[[/o@^-..
#                                         oI8888UU[[[/o@P^:--..
#                                      .@^  YUU[[[/o@^;::---..
#                                    oMP     ^/o@P^;:::---..
#                                 .dMMM    .o@^ ^;::---...
#                                dMMMMMMM@^`       `^^^^
#                               YMMMUP^
#                                ^^
# _____________________________________________________________________________
#
#
# Copyright 2021 Markus Ernst
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.
#
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch.nn as nn
import torch.nn.functional as F
import torch

# custom functions
# -----
import models.metrics as metrics

# -----------------
# Network Constructor
# -----------------

def return_same(x):
    return x

# -----------------
# Simplified Generalized Linear Model (GLM)
# -----------------

class GLM(nn.Module):
    def __init__(self, image_size=1024, input_channels=1, num_targets=10):
        super(GLM, self).__init__()
        self.n_features = image_size*input_channels
        self.fc = nn.Linear(self.n_features, num_targets)

    def forward(self, x):
        seq_len = x.size(1)
        output_list = []
        feature_map = x
        for t in range(seq_len):
            input = x[:, t, :, :, :].view(x.size(0), self.n_features)
            output_list.append(self.fc(input))
        x = torch.stack(output_list, dim=1)
        return x, feature_map
    
    def log_stats(self, writer, step):
        self._varstats2tb(self.fc.weight,
            'linear/fc_weights', writer, step)
        self._varstats2tb(self.fc.bias,
            'linear/fc_bias', writer, step)
            
    def _varstats2tb(self, variable, name, writer, step):
        variable = variable.detach()
        writer.add_scalar(
            'network/{}/mean'.format(name), variable.mean(), step)
        writer.add_scalar(
            'network/{}/std'.format(name), variable.std(), step)
        writer.add_scalar(
            'network/{}/min'.format(name), variable.min(), step)
        writer.add_scalar(
            'network/{}/max'.format(name), variable.max(), step)
        writer.add_scalar(
            'network/{}/median'.format(name), variable.median(), step)
        # writer.add_scalar(
        #     'network/{}/gini'.format(name), metrics.gini_torch(variable), step)


# -----------------
# Class Activation Mapping (CAM)
# -----------------


class CAM(nn.Module):
    def __init__(self, network):
        super(CAM, self).__init__()
        self.network = network
        
    def forward(self, x, topk=3):
        outputs, feature_maps = self.network(x)
        cams = []
        topk_prob_list = []
        topk_arg_list = []
        b, t, c, h, w = feature_maps.size()
        
        
        for timestep in range(t):
            output = outputs[:,timestep,:]
            feature_map = feature_maps[:,timestep,:,:,:]
            
            probs = F.softmax(output, 1)
            prob, args = torch.sort(probs, dim=1, descending=True)
            ## top k class probability
            topk_prob = prob[:,:topk]
            topk_arg = args[:,:topk]

            # generate class activation map
            feature_map = feature_map.view(b, c, h*w).transpose(1, 2)
            fc_weight = nn.Parameter(self.network.fc.weight.t().unsqueeze(0))
            fc_weight = fc_weight.repeat(b, 1, 1)
            cam = torch.bmm(feature_map, fc_weight).transpose(1, 2)
            
            

            ## top k class activation map
            cam = cam.view(b, -1, h, w)
            # top k sorting should be outsourced to the visualization?
            # topk_cam = []
            # for i in range(b):
            #     topk_cam.append(cam[i, topk_arg[i,:],:,:])
            # topk_cam = torch.stack(topk_cam, 0)
            
            cam_upsampled = F.interpolate(cam, 
                                            (x.size(3), x.size(4)), mode='bilinear', align_corners=True)
            
            _,n_classes,fh,fw = cam_upsampled.shape
            cams.append(cam_upsampled)
            topk_prob_list.append(topk_prob)
            topk_arg_list.append(topk_arg)
        
        cams = torch.stack(cams, dim=1)
        topk_prob = torch.stack(topk_prob_list, dim=1)
        topk_arg = torch.stack(topk_arg_list, dim=1)
        return outputs, (cams, topk_prob, topk_arg)



# -----------------
# RC Classes
# -----------------

from torch.nn import init

class RecConvCell(nn.Module):
    """docstring for RCCell."""

    def __init__(self, connectivity, input_channels, output_channels, output_channels_above, kernel_size,padding, bias):
        """
        Initialize RecConvCell cell.

        Parameters
        ----------
        input_channels: int
            Number of channels of input tensor.
        output_channels: int
            Number of channels of the output tensor.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(RecConvCell, self).__init__()
        
        self.connectivity = connectivity
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.output_channels_above = output_channels_above

        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.bottomup = nn.Conv2d(in_channels=self.input_channels,
                                  out_channels=self.output_channels,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

        self.lateral = nn.Conv2d(in_channels=self.output_channels,
                                 out_channels=self.output_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 bias=self.bias)


        self.topdown = nn.Conv2d(in_channels=self.output_channels,
                                 out_channels=self.output_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 bias=self.bias)

        # self.topdown = nn.ConvTranspose2d(in_channels=self.output_channels_above,
        #                                   out_channels=self.output_channels,
        #                                   kernel_size=self.kernel_size,
        #                                   stride=(2,2),
        #                                   padding=1,
        #                                   output_padding=1,
        #                                   groups=1,
        #                                   bias=True,
        #                                   dilation=1,
        #                                   padding_mode='zeros')

        self.bn_b = nn.BatchNorm2d(self.output_channels)
        self.relu_b = nn.ReLU()
        self.bn_l = nn.BatchNorm2d(self.output_channels)
        self.relu_l = nn.ReLU()
        self.bn_t = nn.BatchNorm2d(self.output_channels)
        self.relu_t = nn.ReLU()
        self.bn_all = nn.BatchNorm2d(self.output_channels)

        init.orthogonal_(self.bottomup.weight.data, gain=1)
        init.orthogonal_(self.lateral.weight.data, gain=1)
        init.orthogonal_(self.topdown.weight.data, gain=1)
        init.uniform_(self.bn_b.weight.data, 0.1, 1.0)
        init.constant_(self.bn_b.bias.data, 0.0)
        init.uniform_(self.bn_l.weight.data, 0.1, 1.0)
        init.constant_(self.bn_l.bias.data, 0.0)
        init.uniform_(self.bn_t.weight.data, 0.1, 1.0)
        init.constant_(self.bn_t.bias.data, 0.0)
        init.uniform_(self.bn_all.weight.data, 0.1, 1.0)
        init.constant_(self.bn_all.bias.data, 0.0)


        self.activation = nn.ReLU()
        # TODO: clean this up in order to use it properly
        #self.lrn = nn.LocalResponseNorm(2, alpha=0.0001, beta=0.75, k=1.0)
        self.lrn = return_same

    def forward(self, b_input, l_input, t_input):
        b_conv = self.relu_b(self.bn_b(self.bottomup(b_input)))

        if 'BLT' in self.connectivity:
            l_conv = self.relu_l(self.bn_l(self.lateral(l_input)))
            t_conv = self.relu_t(self.bn_t(self.topdown(t_input)))
            # print(b_conv.shape, l_conv.shape, t_conv.shape)
            next_state = self.lrn(self.activation(b_conv + l_conv + t_conv))
        elif 'BL' in self.connectivity:
            l_conv = self.bn_l(self.lateral(l_input))
            next_state = self.lrn(self.activation(b_conv + l_conv))

        elif 'BT' in self.connectivity:
            t_conv = self.bn_t(self.topdown(t_input))
            next_state = self.lrn(self.activation(b_conv + t_conv))
        else:
            next_state = self.lrn(self.activation(b_conv))
            
        return next_state

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        l = torch.zeros(batch_size,
                            self.output_channels, height, width,
                            device=self.bottomup.weight.device)
        # td = torch.zeros(batch_size,
        #             self.output_channels_above, height//2, width//2,
        #             device=self.topdown.weight.device)
        td = torch.zeros(batch_size,
                    self.output_channels_above, height, width,
                    device=self.topdown.weight.device)
        return (l, td)




class RecConv(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, connectivity, input_dim, hidden_dim, kernel_size,
        num_layers, batch_first=False, padding=1,bias=True, pooling=True):
        super(RecConv, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers + 1)
        if not len(kernel_size) == len(hidden_dim) - 1 == num_layers:
            raise ValueError('Inconsistent list length.')

        # if not(pooling) and num_layers > 1:
        #     raise ValueError('Multiple layers without pooling are not supported')
        # TODO: implement multiple layers without pooling

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.pooling = pooling

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(RecConvCell(connectivity=connectivity,
                                         input_channels=cur_input_dim,
                                         output_channels=self.hidden_dim[i],
                                         output_channels_above=self.hidden_dim[i+1],
                                         kernel_size=self.kernel_size[i],
                                         padding=padding,
                                         bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        if self.pooling:
            self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # TODO: Implement stateful RCNN
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w),
                                             pooling=self.pooling)
        
        seq_len = input_tensor.size(1)
        output_inner = []
    
        for t in range(seq_len):
            layer_output_list = []
            cur_layer_input = input_tensor[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                l, td = hidden_state[layer_idx]
                
                cur_layer_input = self.cell_list[layer_idx](
                    b_input=cur_layer_input,
                    l_input=l,
                    t_input=td)
                layer_output_list.append(cur_layer_input)

                if self.pooling and (layer_idx < (self.num_layers - 1)):
                    cur_layer_input = self.maxpool(cur_layer_input)

            # update hidden states
            for layer_idx in range(self.num_layers - 1):
                hidden_state[layer_idx] = (layer_output_list[layer_idx], layer_output_list[layer_idx+1])
            hidden_state[self.num_layers - 1] = (layer_output_list[self.num_layers - 1], hidden_state[self.num_layers - 1][-1])

            output_inner.append(cur_layer_input)
            # # look at activations
            # activations_inner.append(layer_output_list)
        return torch.stack(output_inner, dim=1)
    
    def return_activations(self, input_tensor, hidden_state=None):

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # TODO: Implement stateful RCNN
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w),
                                             pooling=self.pooling)
        
        seq_len = input_tensor.size(1)
        output_inner = []
        activations_inner = []
        for t in range(seq_len):
            layer_output_list = []
            cur_layer_input = input_tensor[:, t, :, :, :]
            for layer_idx in range(self.num_layers):
                l, td = hidden_state[layer_idx]
                
                cur_layer_input = self.cell_list[layer_idx](
                    b_input=cur_layer_input,
                    l_input=l,
                    t_input=td)
                    
                layer_output_list.append(cur_layer_input)

                if self.pooling and (layer_idx < (self.num_layers - 1)):
                    cur_layer_input = self.maxpool(cur_layer_input)



            # update hidden states
            for layer_idx in range(self.num_layers - 1):
                hidden_state[layer_idx] = (layer_output_list[layer_idx], layer_output_list[layer_idx+1])
            hidden_state[self.num_layers - 1] = (layer_output_list[self.num_layers - 1], hidden_state[self.num_layers - 1][-1])

            output_inner.append(cur_layer_input)
            activations_inner.append(layer_output_list)

        return activations_inner

    def _init_hidden(self, batch_size, image_size, pooling):
        init_states = []
        if pooling:
            init_states.append(self.cell_list[0].init_hidden(batch_size, image_size))
            for i in range(1, self.num_layers):
                init_states.append(self.cell_list[i].init_hidden(batch_size, tuple(dim//(2*i) for dim in image_size)))
        else:
            for i in range(self.num_layers):
                init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class RecConvNet(nn.Module):
    def __init__(self, connectivity, kernel_size, input_channels=1, n_features=32, num_layers=2, num_targets=10):
        super(RecConvNet, self).__init__()
        self.rcnn = RecConv(connectivity, input_channels, n_features, kernel_size, num_layers, batch_first=True)
        self.fc = nn.Linear(n_features, num_targets)
        self.n_features = n_features

    def forward(self, x):
        feature_map = self.rcnn(x)
        x = feature_map.mean(dim=[-2,-1], keepdim=True) #global average pooling
        seq_len = x.size(1)
        output_list = []
        for t in range(seq_len):
            input = x[:, t, :, :, :].view(x.size(0), self.n_features) #8, 8 for osmnist
            output_list.append(self.fc(input))
        x = torch.stack(output_list, dim=1)
        return x, feature_map
    
    def log_stats(self, writer, step):
        for i, layer in enumerate(self.rcnn.cell_list):
            connectivity = layer.connectivity
            self._varstats2tb(layer.bottomup.weight,
                'layer{}/B_weights'.format(i+1), writer, step)
            self._varstats2tb(layer.bottomup.bias,
                'layer{}/B_bias'.format(i+1), writer, step)
            if 'L' in connectivity:
                self._varstats2tb(layer.lateral.weight,
                    'layer{}/L_weights'.format(i+1), writer, step)
                self._varstats2tb(layer.lateral.bias,
                    'layer{}/L_bias'.format(i+1), writer, step)
            if ('T' in connectivity) and (i > 0):
                self._varstats2tb(layer.topdown.weight,
                    'layer{}/T_weights'.format(i+1), writer, step)
                self._varstats2tb(layer.topdown.bias,
                    'layer{}/T_bias'.format(i+1), writer, step)
            
    def _varstats2tb(self, variable, name, writer, step):
        variable = variable.detach()
        writer.add_scalar(
            'network/{}/mean'.format(name), variable.mean(), step)
        writer.add_scalar(
            'network/{}/std'.format(name), variable.std(), step)
        writer.add_scalar(
            'network/{}/min'.format(name), variable.min(), step)
        writer.add_scalar(
            'network/{}/max'.format(name), variable.max(), step)
        writer.add_scalar(
            'network/{}/median'.format(name), variable.median(), step)
        # writer.add_scalar(
        #     'network/{}/gini'.format(name), metrics.gini_torch(variable), step)


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
