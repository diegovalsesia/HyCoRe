"Copied from https://github.com/leymir/hyperbolic-image-embeddings/blob/master/hyptorch/nn.py"

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import geoopt
from modelsnet.ops.manifolds import PoincareBall, Euclidean
#import geoopt.manifolds.stereographic.math as gmath
from geoopt import ManifoldParameter
#from torch.cuda.amp import autocast
import modelsnet.ops.pmath as pmath

MIN_NORM = 1e-15

class HyperbolicMLR(nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """

    def __init__(self, ball_dim, n_classes, c):
        super(HyperbolicMLR, self).__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.reset_parameters()

    def forward(self, x, c=None):
        if c is None:
            c = torch.as_tensor(self.c).type_as(x)
        else:
            c = torch.as_tensor(c).type_as(x)
        p_vals_poincare = pmath.expmap0(self.p_vals, c=c)
        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor
        logits = pmath._hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits

    def extra_repr(self):
        return "Poincare ball dim={}, n_classes={}, c={}".format(
            self.ball_dim, self.n_classes, self.c
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))


class MobiusMLR(nn.Module):
    """
    Multinomial logistic regression in the Poincare Ball
    It is based on formulating logits as distances to margin hyperplanes.
    In Euclidean space, hyperplanes can be specified with a point of origin
    and a normal vector. The analogous notion in hyperbolic space for a
    point $p \in \mathbb{D}^n$ and
    $a \in T_{p} \mathbb{D}^n \backslash \{0\}$ would be the union of all
    geodesics passing through $p$ and orthogonal to $a$. Given $K$ classes
    and $k \in \{1,...,K\}$, $p_k \in \mathbb{D}^n$,
    $a_k \in T_{p_k} \mathbb{D}^n \backslash \{0\}$, the formula for the
    hyperbolic MLR is:
    \begin{equation}
        p(y=k|x) f\left(\lambda_{p_k} \|a_k\| \operatorname{sinh}^{-1} \left(\frac{2 \langle -p_k \oplus x, a_k\rangle}
                {(1 - \| -p_k \oplus x \|^2)\|a_k\|} \right) \right)
    \end{equation}
    """

    def __init__(self, in_features, out_features, c=1.0):
        """
        :param in_features: number of dimensions of the input
        :param out_features: number of classes
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = PoincareBall(c=c, dim= in_features)
        points = torch.randn(out_features, in_features) * 1e-5
        points = self.ball.expmap0(points)	#, k=self.ball.k)
        self.p_k = ManifoldParameter(points, manifold=self.ball)

        tangent = torch.Tensor(out_features, in_features)
        stdv = (6 / (out_features + in_features)) ** 0.5  # xavier uniform
        torch.nn.init.uniform_(tangent, -stdv, stdv)
        self.a_k = torch.nn.Parameter(tangent)

    def forward(self, input):
        """
        :param input: batch x space_dim: points (features) in the Poincaré ball
        :return: batch x classes: logit of probabilities for 'out_features' classes
        """
        input = input.unsqueeze(-2)     # batch x aux x space_dim
        distance, a_norm = self._dist2plane(x=input, p=self.p_k, a=self.a_k, c=self.ball.c, k=-self.ball.c, signed=True)
        result = 2 * a_norm * distance
        return result

    def _dist2plane(self, x, a, p, c, k, keepdim: bool = False, signed: bool = False, dim: int = -1):
        """
        Taken from geoopt and corrected so it returns a_norm and this value does not have to be calculated twice
        """
        sqrt_c = c ** 0.5
        minus_p_plus_x = pmath.mobius_add(-p, x, c=c)	#, dim=dim)
        mpx_sqnorm = minus_p_plus_x.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
        mpx_dot_a = (minus_p_plus_x * a).sum(dim=dim, keepdim=keepdim)
        if not signed:
            mpx_dot_a = mpx_dot_a.abs()
        a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
        num = 2 * sqrt_c * mpx_dot_a
        denom = (1 - c * mpx_sqnorm) * a_norm
        return pmath.arsinh(num / denom.clamp_min(MIN_NORM)) / sqrt_c, a_norm

    def extra_repr(self):
        return "in_features={in_features}, out_features={out_features}".format(**self.__dict__) + f" k={self.ball.k}"




class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = pmath.mobius_matvec(self.weight, x, c=c)
        if self.bias is None:
            return pmath.project(mv, c=c)
        else:
            bias = pmath.expmap0(self.bias, c=c)
            return pmath.project(pmath.mobius_add(mv, bias), c=c)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )


class ConcatPoincareLayer(nn.Module):
    def __init__(self, d1, d2, d_out, c):
        super(ConcatPoincareLayer, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out

        self.l1 = HypLinear(d1, d_out, bias=False, c=c)
        self.l2 = HypLinear(d2, d_out, bias=False, c=c)
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.mobius_add(self.l1(x1), self.l2(x2), c=c)

    def extra_repr(self):
        return "dims {} and {} ---> dim {}".format(self.d1, self.d2, self.d_out)


class HyperbolicDistanceLayer(nn.Module):
    def __init__(self, c):
        super(HyperbolicDistanceLayer, self).__init__()
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.dist(x1, x2, c=c, keepdim=True)

    def extra_repr(self):
        return "c={}".format(self.c)


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


class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):

        super(FromPoincare, self).__init__()

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

        self.train_c = train_c
        self.train_x = train_x

    def forward(self, x):
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return pmath.logmap(xp, x, c=self.c)
        return pmath.logmap0(x, c=self.c)

    def extra_repr(self):
        return "train_c={}, train_x={}".format(self.train_c, self.train_x)



class Distance2PoincareHyperplanes(torch.nn.Module):
    n = 0
    # 1D, 2D versions of this class ara available with a one line change
    # class Distance2PoincareHyperplanes2d(Distance2PoincareHyperplanes):
    #     n = 2

    def __init__(
        self,
        plane_shape: int,
        num_planes: int,
        signed=True,
        squared=False,
        *,
        ball,
        std=1.0,
    ):
        super().__init__()
        self.signed = signed
        self.squared = squared
        # Do not forget to save Manifold instance to the Module
        self.ball = ball
        self.plane_shape = geoopt.utils.size2shape(plane_shape)
        self.num_planes = num_planes

        # In a layer we create Manifold Parameters in the same way we do it for
        # regular pytorch Parameters, there is no difference. But geoopt optimizer
        # will recognize the manifold and adjust to it
        self.points = geoopt.ManifoldParameter(
            torch.empty(num_planes, plane_shape), manifold=self.ball
        )
        self.std = std
        # following best practives, a separate method to reset parameters
        self.reset_parameters()

    def forward(self, input):
        input_p = input.unsqueeze(-self.n - 1)
        points = self.points.permute(1, 0)
        points = points.view(points.shape + (1,) * self.n)

        distance = self.ball.dist2plane(
            x=input_p, p=points, a=points, signed=self.signed, dim=-self.n - 2
        )
        if self.squared and self.signed:
            sign = distance.sign()
            distance = distance ** 2 * sign
        elif self.squared:
            distance = distance ** 2
        return distance

    def extra_repr(self):
        return (
            "plane_shape={plane_shape}, "
            "num_planes={num_planes}, "
            .format(**self.__dict__)
        )

    @torch.no_grad()
    def reset_parameters(self):
        direction = torch.randn_like(self.points)
        direction /= direction.norm(dim=-1, keepdim=True)
        distance = torch.empty_like(self.points[..., 0]).normal_(std=self.std)
        self.points.set_(self.ball.expmap0(direction * distance.unsqueeze(-1)))
