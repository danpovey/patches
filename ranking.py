import torch
from torch import nn
from torch import Tensor
from typing import Tuple
import random

# differentiable rank (easy-to-implement version)

try:
    from scaling import FloatLike
except:
    FloatLike = float


class Sort(nn.Module):
    def __init__(self,
                 stddev_factor: float = 1.0,
                 use_random_rank: bool = True):
        """
        Args:
          stddev_factor: you can set this to a value larger than 1.0 to prevent large
            gradients that might happen if points are too "spaced out".  We'll
            occasionally print the minimum
            appropriately normalized gradient, to see whether it is too small.

          use_random_rank: if True, return the randomized rank that are used to make the
            ranks differentiable. This may be more mathematically correct than using
            the deterministic ranks, for learning the sorting, but probably won't make
            much difference.
        """
        super().__init__()
        self.stddev_factor = stddev_factor
        self.use_random_rank = use_random_rank

    def forward(self,
                x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward function that differentiably sorts elements of input tensor x,
        interpreted as scores, and returns the indexes together with a
        differentiable float version of the ranks [ 0.0, 1.0, 2.0, ... ].
        Caution: the algorithm used in training time uses O(N^2) memory and
        compute, where N is the number of things to be sorted; but with
        a good constant of proportionality, especially on GPU.

          Args:
            x: (B, N)  where N is the dimension we sort on and B is interpreted as the
                   batch size.
           Returns:
            indexes: (B, N), a LongTensor containing a permutation of [0, 1, 2, .. N-1].
               rank: (B, N), a Tensor containing [ 0.0, 1.0, 2.0, ... N-1 ] (not permuted),
                     which is differentiable (the derivative reflects the change of the
                     expected rank w.r.t. the elements of x; the ranking ia nondeterministic).
        """
        (batch_size, N) = x.shape
        if self.training or x.requires_grad:
            # clamp x to [-1..1]; this is to force some overlap between the
            # Gaussian distributions.
            x = x.clamp(min=-1, max=1)
            # 'stddev' is the stddev of the distributions of each value x for purposes
            # of computing the differentiable rankings, up to some constant that depends on the
            # math, that I haven't figured out yet.

            stddev = self.stddev_factor / N

            # diffs: (*, N, N); diffs[..,i,j] equals x[...,i] - x[...,j] so it is
            # positive of x[..., i] > x[..., j], therefore diffs.sum(-1)[...,i] would equal
            # the rank of x[..., i].  We'll use a differentiable form of this
            diffs = (x.unsqueeze(-1) - x.unsqueeze(-2))
            # Here, if the distribution were Gaussian with variance "stddev" we would do:
            # diffs = gaussint(diffs / (stddev * math.sqrt(2.0)))
            # where gaussint(x) = 0.5 + 0.5 * x.erf().
            # If you put into wolframalpha:
            # 0.5 + 0.5 * erf(x / sqrt(2)), sigmoid(x*2)
            # you'll see that gaussint(x/sqrt(2)) looks very similar to sigmoid(x*2); the sigmoid
            # is easier to compute. (the 2 is just an arbitrary factor to make the plots match).
            # We drop the factor of 2.0, though, as it can be absorbed into stddev_factor.
            # In principle we can work backward from the .sigmoid(), treating it as the
            # integral of a distribution and then undo the convolution-with-itself operation*
            # to find the distribution of the x values that it corresponds to.  The point is
            # that it corresponds to some distribution
            # (*could do this by doing square root in the space of fourier coefficients, which
            # will work because there are no negative coefficients due to symmetry and convexity).

            def gaussint(x):  # gaussian integral of x
                return 0.5 + (0.5 * x.erf())
            # the difference between two standard normal Gaussians has variance
            # 2.0, and stddev 1.0.
            diffs = gaussint(diffs / ((2.0 ** 0.5) * stddev))
            # diff_ranks: differentiable ranks.
            diff_ranks = diffs.sum(dim=-1)

            if random.random() < 0.01:
                self._print_diagnostics(x)

            if self.use_random_rank:
                x = x + torch.randn_like(x) * stddev

            _values,indexes = x.sort(dim=-1)

            diff_ranks = torch.gather(diff_ranks, dim=-1, index=indexes)
            # diff_ranks will now be roughly (if use_random_rank) or exactly in sorted order.
            arange = torch.arange(N, device=x.device).to(torch.float)

            #print(f"diff_ranks[0] = {diff_ranks}, vs. arange = {arange}")
            diff_ranks = diff_ranks + (arange - diff_ranks).detach()
            # diff_ranks will now be numerically equal to 'arange', but will propagate
            # its derivative back to 'diff_ranks'.
            return indexes, diff_ranks
        else:
            # Caution: if we are doing something in eval mode where we actually
            # want gradients w.r.t parameters for some reason, this is not correct.
            _values,indexes = x.sort(dim=-1)
            arange = torch.arange(N, device=x.device).to(torch.float)
            arange = arange.unsqueeze(0).expand(batch_size, N)
            return indexes, arange

    def _print_diagnostics(self,
                           x: Tensor):
        # note: at this point, we have already clamped x to -1..1
        N = x.shape[-1]
        stddev = self.stddev_factor / N
        values, indexes = x.sort(dim=-1)

        lower_clamp = (x <= -1).to(torch.float).mean()
        upper_clamp = (x >= 1).to(torch.float).mean()


        num_discard = N // 8
        values = values[:, num_discard:-num_discard]  # discard the bottom and
                                                      # top eights of the
                                                      # distribution

        # diffs will all be >= 0.
        diffs = values[..., 1:] - values[..., :-1]
        diff_values, _indexes =  diffs.sort(dim=-1)

        largest_indexes = torch.gather(indexes, dim=-1, index=_indexes[:, -1:]).flatten()

        print(f"upper_clamped={upper_clamp}, lower_clamped={lower_clamp}, diffs largest={diff_values[:,-1]} vs. stddev {stddev}, largest_indexes={largest_indexes}, diffs mean={diff_values.mean()}")


class Subset(nn.Module):
    def __init__(self,
                 num_channels: int,
                 noised_fraction: float = 0.1,
                 stddev_factor: float = 4.0,
                 use_random_rank: bool = True,
                 random_fraction: FloatLike = 0.1):
        """
        Selects a subset of the input embeddings (the size of the subset is
        specified in forward().  Those near the boundary are replaced with noise.

        Args:
               num_channels:  the number of input channels
              noised_fraction: the fraction of the input embeddings that have
                noise added to them; a too-small value will make the gradients
                too peaky and things prior to this module harder to learn.
          random_fraction: the fraction of scores that are to be from random
            locations, in training mode.  We will only apply random_fraction for 75%
            of the time (decided per batch-element), so that it learns to be compatible
            with test time when random_fraction is not applied.  The purpose of this is
            to help training get started.
        """
        super().__init__()
        self.to_scores = nn.Linear(num_channels, 1)
        self.noised_fraction = noised_fraction
        self.sort = Sort(stddev_factor=stddev_factor, use_random_rank=use_random_rank)
        self.random_fraction = random_fraction

    def forward(self,
                x: Tensor,
                N: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward function that selects some elements of x.  Args:
          x: the embeddings to select from, of shape (batch_size, num_embeddings, num_channels))
           N: the number of embeddings from 'num_embeddings' to keep.

        Returns: (indexes, y, weights), where:
             indexes: (batch_size, N), the indexes of the chosen embeddings
                   y: (batch_size, N, num_channels), the selected features
             weights: (batch_size, N, 1), weights between 0 and 1 that will
                    only be != 1 for a fraction "noised_fraction" of the weights.
        """
        (batch_size, num_embeddings, num_channels) = x.shape
        x = self.to_scores(x).squeeze(-1)
        if self.training:
            x = self._randomize_some(x)
        # scores: (batch_size, num_embeddings)

        indexes, ranks = self.sort(scores)
        # indexes, ranks: (batch_size, num_embeddings)
        # indexes are integers that index the embedding dimension
        # ranks are [0.0, 1.0, 2.0... ], but differentiable.

        indexes = indexes[:, -N:]

        x = torch.gather(x, dim=1, index=indexes.unsqueeze(-1).expand(batch_size, N, num_channels))

        # add noise.  For now, do this even in test mode, in case the model comes to rely on.
        num_discarded = num_embeddings - N
        x_scale = ((ranks[:, -N:].unsqueeze(-1) - num_discarded) / (self.noised_fraction * N)).clamp(max=1.0)
        # x_scale: (batch_size, N, 1)
        noise_scale = 1.0 - x_scale

        eps = 1.0e-05
        # include the rms of each embedding vector in noise_scale, to avoid x learning to
        # get large to defeat th enoise.
        noise_scale = noise_scale * ((x ** 2).mean(dim=2, keepdim=True) + eps) ** 0.5

        y = (x * x_scale) + (torch.randn_like(x) * noise_scale)

        return indexes, y, x_scale



    def _randomize_some(self, x: Tensor) -> Tensor:
        """
        Randomizes a fraction of the elements of x (scores) by replacing them with other elements
        """
        random_fraction = float(self.random_fraction)
        if random_fraction == 0.0:
            return
        mask = torch.rand_like(x) < random_fraction  # True for things we'll replace.
        batch_mask = torch.rand_like(x[:, :1]) < 0.75
        mask = torch.logical_and(mask, batch_mask)
        # for 25% of batch elements, don't do randomization, so the model learns to deal with
        # the test-time condition where there is no randomization.

        x_rand = x.flip(dims=(0,1))
        # for the elements where 'mask' is true, return x_rand, which is from a
        # pseudo-random location, if it's larger than x; otherwise, return x.
        # we'll never let the randomization turn things *off* only on, hence the
        # "max".
        return torch.where(mask, torch.max(x, x_rand), x)




def _test_sort_numbers():
    num_channels = 128
    model = Subset(num_channels=num_channels)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=2e-04)

    for step in range(10000):
        batch_size = 10
        num_embeddings = 256
        N = 32  # keep N embeddings

        x = torch.randn(batch_size, num_embeddings, num_channels)

        _indexes, y, _weights = model(x, N)

        # want the N largest elements of the 1st dim of x
        loss = - y[..., 0].sum()

        # ref_loss is the best we could have done.

        values, _indexes2 = x[..., 0].sort(dim=-1)
        ref_loss = -values[:,-N:].sum()

        if step % 100 == 0:
            den = N * batch_size

            # get loss in eval mode
            model.eval()
            _indexes, y, _weights = model(x, N)
            model.train()
            # want the N largest elements of the 1st dim of x
            loss_eval = - y[..., 0].sum()
            print(f"step={step}, loss={loss.item()/den}, loss_eval={loss_eval.item()/den}, ref_loss={ref_loss.item()/den}, to_scores={model.to_scores.weight}")

        loss.backward()
        optim.step()
        optim.zero_grad()


if __name__ == '__main__':
    _test_sort_numbers()
