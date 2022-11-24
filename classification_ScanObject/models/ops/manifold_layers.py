import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from modelsnet.manifolds import PoincareBall, Euclidean
from geoopt import ManifoldParameter


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c
        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c

        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):

        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)



class RiemannianLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold, over_param, weight_norm):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold

        self._weight = Parameter(torch.Tensor(out_features, in_features))
        self.over_param = over_param
        self.weight_norm = weight_norm
        if self.over_param:
            self._bias = ManifoldParameter(torch.Tensor(out_features, in_features), manifold=manifold)
        else:
            self._bias = Parameter(torch.Tensor(out_features, 1))
        self.reset_parameters()

    @property
    def weight(self):
        return self.manifold.transp0(self.bias, self._weight) # weight \in T_0 => weight \in T_bias

    @property
    def bias(self):
        if self.over_param:
            return self._bias
        else:
            return self.manifold.expmap0(self._weight * self._bias) # reparameterisation of a point on the manifold

    def reset_parameters(self):
        init.kaiming_normal_(self._weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
        bound = 4 / math.sqrt(fan_in)
        init.uniform_(self._bias, -bound, bound)
        if self.over_param:
            with torch.no_grad(): self._bias.set_(self.manifold.expmap0(self._bias))


class GeodesicLayer(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(GeodesicLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        input = input.unsqueeze(-2).expand(*input.shape[:-(len(input.shape) - 2)], self.out_features, self.in_features)
        res = self.manifold.normdist2plane(input, self.bias, self.weight,
                                               signed=True, norm=self.weight_norm)
        return res


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super(Linear, self).__init__(
            in_features,
            out_features,
        )


class MobiusLayer(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(MobiusLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        res = self.manifold.mobius_matvec(self.weight, input)
        return res


class ExpZero(nn.Module):
    def __init__(self, manifold):
        super(ExpZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.expmap0(input)


class LogZero(nn.Module):
    def __init__(self, manifold):
        super(LogZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.logmap0(input)

################################################################################
#
# Hyperbolic VAE encoders/decoders
#
################################################################################

class GyroplaneConvLayer(nn.Module):
    """
    Gyroplane 3D convolutional layer based on GeodesicLayer
    Maps Poincare ball to Euclidean space of dim filters
    """
    def __init__(self, in_features, out_channels, kernel_size, manifold):
        super(GyroplaneConvLayer, self).__init__()
        
        self.in_features = in_features
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.manifold = manifold

        self.gyroplane_layer = GeodesicLayer(self.in_features, self.out_channels, self.manifold)

    def forward(self, input):
        """
        input: [batch_size, in_channels, width, height, depth, manifold_dim] # ([128, 1, 2, 2, 2])
        output: [batch_size, out_channels, width, height, depth]
        """ 
        
        #input = torch.zeros([8 * 8 * 8, 128, 2], dtype=torch.float32, device='cuda') # DELETE THIS & CHANGE KERNEL SIZE
        
        sampled_dim = input.shape[0]
        batch = input.shape[1]
        manifold_dim = input.shape[2]
        size = self.kernel_size
        s = size // 2
        
        input = input.permute(1, 0, 2).view(batch,
                           int(round(sampled_dim**(1/3.0))),
                           int(round(sampled_dim**(1/3.0))),
                           int(round(sampled_dim**(1/3.0))),
                           manifold_dim)
        
        padded_input = torch.zeros((batch, 
                                    input.shape[1] + 2 * s, 
                                    input.shape[2] + 2 * s, 
                                    input.shape[3] + 2 * s, 
                                    manifold_dim), device='cuda')
        padded_input[:, s:input.shape[1]+s, s:input.shape[2]+s, s:input.shape[3]+s, :] = input
        input = padded_input
        
        width = input.shape[1]
        height = input.shape[2]
        depth = input.shape[3]
                
        combined_output = []
        for i in range(s, width - s):
            for j in range(s, height - s):
                for k in range(s, depth - s):
                    patch = input[:, i-s: i+s+1, j-s: j+s+1, k-s: k-s+1, :]
                    patch = patch.reshape(batch, size * size * size, manifold_dim)

                    layer_output = self.gyroplane_layer(patch)
                    layer_output = torch.sum(layer_output, dim = 1)
                    
                    combined_output.append(layer_output)
        combined_output = torch.stack(combined_output).permute(1, 2, 0).view(batch,
                                                                             -1,
                                                                             width - 2 * s,
                                                                             height - 2 * s,
                                                                             depth - 2 * s)        

        return combined_output
