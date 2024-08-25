import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Optional
import random
from ranking import Subset

try:
    from scaling import FloatLike
except:
    FloatLike = float



def _get_patch_indexes(tot_size: Tuple[int, int],
                       required_size: Tuple[int, int],
                       batch_size: int,
                       h_offset: Optional[Tensor],
                       w_offset: Optional[Tensor],
                       device) -> Tensor:
    """
    Return the patch indexes (numbers between 0 and tot_size[0]*tot_size[1]
    that correspond to a sub-image of size "required_size" of the full image,
    with the given offsets.

    Args:
         tot_size: maximimum potential size of image MEASURED IN PATHCES as (H, W): (e.g. to define position-embedding sizes
     required_size: size of sub-image required, MEASURED IN PATCHES: (h, w)
          h_offset: if provided, a Tensor of shape (batch_size,) giving offsets from
                    the top MEASURED IN PATCHES
          w_offset: if provided, a Tensor of shape (batch_size,) giving offsets from
                    the left of the image MEASURED IN PATCHES.   h_offset and w_offset
                     should either both be provided, or neither.

    Returns:
      If h_offset and w_offset is provided:
       a LongTensor of shape (batch_size, required_size[0]*required_size[1]), containing indexes
       0 <= i < tot_size[0]*tot_size[1]
     Otherwise: of shape (1, required_size[0]*required_size[1],).
    """
    H, W = tot_size  # size in patches
    h, w = required_size  # size in patches
    arange = torch.arange(max(h, w),
                          device=device)


    arange_h = torch.arange(h, device=device)
    arange_w = torch.arange(w, device=device)


    if h_offset is not None:
        assert h_offset.ndim == 1
        assert batch_size == h_offset.shape[0]
        arange_h = arange_h + h_offset.unsqueeze(-1)
        arange_w = arange_w + w_offset.unsqueeze(-1)

    ans = (arange_h * W).unsqueeze(-1) + arange_w.unsqueeze(-2)

    return ans.reshape(-1, h * w).expand(batch_size, h * w)


def _patchify(x: Tensor,
              patch_size: Tuple[int, int]) -> Tensor:
    """
    Turn image x in standard format for image storage, (N, H, W, C),
    into patches in format (batch_size, num_embed, patch_h * patch_w * img_channels)

    Args:
        x: (N, C_img, H, W), the input batch of images
      patch_size: the patch size (h, w); must divide the height and width of
           x exactly.
     Returns:
        Image x turned into patches, of shape:
           (N, (H * W) / (h * w), h * w * C_img)
    """
    (N, H, W, C_img) = x.shape
    (h, w) = patch_size
    x = x.reshape(N, H // h, h, W // w, w, C_img)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(N, (H // h) * (W // w), h * w * C_img)
    return x

def _unpatchify(emb: Tensor,
                img_size: Tuple[int, int],
                patch_size: Tuple[int, int]) -> Tensor:
    """
    Turn patches back into an image in (N, H, W, C) format, from
    the format (batch_size, num_embed, patch_h * patch_w * img_channels); the
    reverse of _patchify.

    Args:
       emb: (N, num_embed, emb_dim), interpreted as:
            (N,  (N, (H * W) / (h * w), h * w * C_img), the input batch of embeddings.
      img_size: the size (H, W) of the image that we are trying to reconstruct.
             (We cannot figure this out from emb and patch_size because of ambiguity
             about the aspect ratio of the image).
     patch_size: the size (h, w) of the patches.

        x: (N, C_img, H, W), the input batch of images
      patch_size: the patch size (h, w); must divide the height and width of
           x exactly.
     Returns:
        Image x turned into patches, of shape:
           (N, (H * W) / (h * w), h * w * C_img)
    """
    (N, num_embed, emb_dim) = emb.shape
    (H, W) = img_size
    (h, w) = patch_size
    assert emb_dim % (h * w) == 0
    C_img = emb_dim // (h * w)

    x = emb.reshape(N, (H // h), (W // w), h, w, C_img)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(N, H, W, C_img)
    return x


def _get_relative_indexes(available: Tensor,
                          required: Tensor,
                          N: int,
                          support_available_out_of_range: bool = False,
                          support_required_out_of_range: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Args:
      available: (batch_size, A), a LongTensor containing a list of indexes 0 <= i < N in
        arbitrary order without repeated values, saying which patches are available (e.g
        which ones we have computed embeddings for).  if support_out_of_range == True,
        IT MAY ALSO CONTAIN OTHER VALUES, i.e. outside the interval [0..N-1], and these
        will be ignored and never matched.
      required: (batch_size, B), a LongTensor containing a list of indexes 0 <= i < N without
        repeated values, saying which index values we "want" to be find in the list "available".
        E.g. these will correspond to a subset of positions in a "template image" of largest
        possible size N.
      N: defines the range in which the indexes in 'required' must lie; if support_out_of_range
          == False,  also defines the range in which the indexes in 'available' must lie.
      support_available_out_of_range: True if 'available' may have values outside [0..N-1], to be ignored.
      support_required_out_of_range: True if 'required' may have values outside [0..N-1], to be ignored.
            At most one of support_available_out_of_range and support_available_out_of_range may be
            False
    Returns:
          (indexes, mask), where:
        indexes: a LongTensor of shape (batch_size, B) containing indexes
           0 <= i < A, showing the position in 'available'  that corresponds
           to this index in 'required', if it exists; and if not, an arbitrary
           index into 'available'.
        mask: a Tensor of shape (batch_size, B) containing 1.0 for positions in
          'required' for which we found a matching index in 'available', and 0.0
          otherwise.
    """
    (batch_size, A) = available.shape
    (_batch_size, B) = required.shape
    assert _batch_size == batch_size
    device = available.device

    arange = torch.arange(max(A, B), device=device).unsqueeze(0).expand(batch_size, max(A, B))
    arange_A = arange[:, :A]
    arange_B = arange[:, :B]

    assert not (support_available_out_of_range and support_required_out_of_range)
    support_out_of_range = support_available_out_of_range or support_required_out_of_range
    def limit(indexes):
        # replace indexes less than zero with N, then clamp to max=N.
        indexes = torch.where(indexes >= 0, indexes, torch.full((1, 1), N).expand_as(indexes))
        return indexes.clamp_(max=N)
    if support_available_out_of_range:
        available = limit(available)
    if support_required_out_of_range:
        required = limit(required)


    # a_indexes will be for each value 0 <= n <= N, the position in 'available'
    # contains that value n if it exists, else -1.

    if support_out_of_range:
        # TODO: use N + 16 as the size instead of N + 1, to retain memory cache alignment.
        a_indexes_N = torch.full((batch_size, N + 16), -1, device=device)[:, :N+1]
    else:
        a_indexes_N = torch.full((batch_size, N), -1, device=device)
    a_indexes_N.scatter_(dim=1, index=available, src=arange_A)

    a_indexes_B1 = torch.gather(a_indexes_N, dim=1, index=required)

    # assign "default src indexes" for positions in "required" where nothing was available.
    # This is because torch gather and scatter do not allow an "invalid index" like -1.
    a_indexes_B2 = arange_B
    if B > A:
        a_indexes_B2 = a_indexes_B2 % A

    mask = (a_indexes_B1 >= 0)
    a_indexes_B = torch.where(mask, a_indexes_B1, a_indexes_B2)

    return a_indexes_B, mask.to(torch.float)


def _subsample(img: Tensor,
               subsample_factor: int) -> Tensor:
    """ Subsamples image 'img' by simple averaging over square
      patches.  Requires that subsample_factor exactly divide the image
             height and width.
    Args:
          img:  (batch_size, height, width, img_channels)
      Returns:
          subsampled image of shape (batch_size, height // subsample_factor,
                                     width // subsample_factor, img_channels)
    """
    if subsample_factor == 1:
        return img
    (batch_size, height, width, img_channels) = img.shape
    s = subsample_factor
    img = img.reshape(batch_size, height // s, s, width // s, s, img_channels)
    return img.mean(dim=(2,4))


def _upsample(img: Tensor,
              upsample_factor: int) -> Tensor:
    """ Up image 'img' by repeating values over square patches.

    Args:
          img:  (batch_size, height, width, img_channels)
      Returns:
          upsampled image of shape (batch_size, height * upsample_factor,
                                     width * upsample_factor, img_channels)
    """
    if upsample_factor == 1:
        return img
    (batch_size, height, width, img_channels) = img.shape
    u = upsample_factor
    img = img.reshape(batch_size, height, 1, width, 1, img_channels)
    img = img.expand(batch_size, height, u, width, u, img_channels)
    return img.reshape(batch_size, height * u, width * u, img_channels)


class AbsolutePosEmb(nn.Module):
    pass


class PatchExtractor(nn.Module):
    # converts images into patches.  Has a maximum image size but can accept
    # smaller images than the maximum.
    def __init__(self,
                  img_channels: int,
                  num_channels: int,
                  max_size: Tuple[int, int] = (512, 512),
                  patch_size: Tuple[int, int] = (4, 4)):
        super().__init__()
        assert len(max_size) == 2 and len(patch_size) == 2
        assert max_size[0] % patch_size[0] == 0 and max_size[1] % patch_size[1] == 0
        num_indexes = (max_size[0] * max_size[1]) // (patch_size[0] * patch_size[1])
        self.max_size = max_size
        self.patch_size = patch_size
        self.num_patches = (max_size[0] // patch_size[0], max_size[1] // patch_size[1])
        self.num_indexes = num_indexes
        self.to_embed = nn.Linear(patch_size[0] * patch_size[1] * img_channels,
                                  num_channels)

    def forward(self,
                x: Tensor,
                h_offset: Optional[Tensor] = None,
                w_offset: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
              x:  (N, H, W, C_img), e.g. C_img == 3, the input image; (H, W) must
                  be a multiple of self.patch_size.

       h_offset, w_offset: height and weight offsets, of shape (N,) i.e. batch size,
              containing offsets IN PIXELS of how the image "x" should be offset
              w.r.t the top-left corner of the "template image" of size "max_size".
              The default None is equivalent to all zeros, i.e. top-left location.

        Returns (emb, indexes), where:
             emb: (N, num_embed, C_emb)
          indexes: LongTensor of shape (N, num_embed), contain numbers 0 <= i < self.num_indexes
        """
        (batch_size, img_height, img_width, img_channels) = x.shape

        patches = _patchify(x, self.patch_size)
        # patches: (batch_size, num_patches, patch_h*patch_w*img_channels)
        (patch_h, patch_w) = self.patch_size
        (max_h, max_w) = self.max_size
        max_npatches = self.num_patches
        img_npatches = (img_height // patch_h, img_width // patch_w)
        if h_offset is not None:
            if random.random() < 0.01:
                assert torch.all(h_offset % patch_h == 0)
                assert torch.all(w_offset % patch_w == 0)
            h_offset = h_offset // patch_h
            w_offset = w_offset // patch_w
        patch_indexes = _get_patch_indexes(max_npatches, img_npatches,
                                           batch_size,
                                           h_offset, w_offset, x.device)
        # patches: (batch_size, num_embeddings, patch_h*patch_w*img_channels)
        # patch_indexes: (batch_size, num_embeddings)
        patches = self.to_embed(patches)
        return patches, patch_indexes

    def forward_indexes(self,
                        x: Tensor,
                        indexes: Tensor,
                        h_offset: Optional[Tensor] = None,
                        w_offset: Optional[Tensor] = None) -> Tensor:
        """
        Version of forward function that accepts 'indexes' so that we will extract
        perhaps only a subset of the available patches in this image.

        Args:
              x:  (N, H, W, C_img), e.g. C_img == 3, the input image; (H, W) must
                  be a multiple of self.patch_size.
        indexes: a LongTensor of shape (N, num_embed), the selected subset of indexes,
            not all of which have to be "valid".  We define a valid index as
            an index 0 <= i < self.num_indexes.  Valid indexes must not be repeated,
            and must be within the patch of the template image that depends on the
            size of x and the values of h_offset and w_offset, as would be
            returned by self.forward().
       h_offset, w_offset: height and weight offsets, each of shape (N,) i.e. batch size,
              containing offsets IN PIXELS of how the image "x" should be offset
              w.r.t the top-left corner of the "template image" of size "max_size".
              The default None is equivalent to all zeros, i.e. top-left location.

        Returns: (emb, mask), where:
              emb: (N, num_embed, num_channels), containing the embeddings (those at
                 positions where indexes < 0 or indexes >= self.num_indexes will be 0.)
             mask: (N, num_embed), a Tensor of floats of shape (N, num_embed) with
                 1.0 at "valid positions" corresponding to the elements
                 0 <= i < self.num_indexes of indexes; and 0.0 elsewhere.
          """
        (batch_size, img_height, img_width, img_channels) = x.shape
        is_valid = torch.logical_and(indexes >= 0, indexes < self.num_indexes)

        emb, patch_indexes = self(x, h_offset, w_offset)

        rel_indexes, mask = _get_relative_indexes(patch_indexes, indexes, self.num_indexes,
                                                  support_required_out_of_range=True)

        emb = torch.gather(emb, dim=1, index=rel_indexes[..., None].expand(*rel_indexes.shape, emb.shape[-1]))
        emb = emb * mask[..., None]
        return emb, mask


class ImageReconstructor(nn.Module):
    # Converts embeddings back into images, like the reverse of PatchExtractor.
    # Because we will later want to combine these images across different
    # resolutions of patches, for each image pixel it also outputs a 'weight'
    # between 0 and 1 which reflects to what extent we had valid input for the patch it was
    # part of.

    # converts from patches to images.
    # into patches.  Has a maximum size but is flexible about
    # the image size.
    def __init__(self,
                 img_channels: int,
                 emb_channels: int,
                 max_size: Tuple[int, int] = (512, 512),
                 patch_size: Tuple[int, int] = (4, 4)):
        super().__init__()
        self.max_size = max_size
        self.patch_size = patch_size
        num_indexes = (max_size[0] * max_size[1]) // (patch_size[0] * patch_size[1])
        self.num_indexes = num_indexes

        self.to_img = nn.Linear(emb_channels,
                                img_channels * patch_size[0] * patch_size[1])

    def forward(self,
                emb: Tensor,
                indexes: Tensor,
                weights: Tensor,
                img_size: Tuple[int, int],
                h_offset: Tensor = None,
                w_offset: Tensor = None,
                support_out_of_range: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Args:
              emb:  (N, num_embed, C_emb), the input embeddings; N is batch size.
           indexes: (N, num_embed), the indexes of the embeddings, between 0 and
                self.num_indexes - 1 inclusive.
           weights: (N, num_embed), the weights corresponding to each of
                these embeddings, to ensure differentiablility, as returned by
                class Subset()'s forward function.  Implicitly, non-present
                embeddings have weight zero.  The weights will be processed
                and returned at the pixel level.
             img_size: the size of the image x that we are trying to reconstruct,
                in pixels
           h_offset, w_offset:  offsets in pixels into the template
               image of size self.max_size.  Must be a multiple of
               the patch size.

        Returns (x, weight), where:
             c: (N, H, W, C_img)  is the reconstructed part of the image.
        weight: (N, H, W, 1) consists of weights between 0 and 1 which
             say to what extent this weight is "valid".  (some intermediate
             values are needed for differentiability during training)
        """
        (patch_h, patch_w) = self.patch_size
        (max_h, max_w) = self.max_size
        (img_h, img_w) = img_size
        max_npatches = (max_h // patch_h, max_w // patch_w)
        img_npatches = (img_h // patch_h, img_w // patch_w)
        batch_size = emb.shape[0]

        if h_offset is not None:
            h_offset = h_offset // patch_h
            w_offset = w_offset // patch_w
        patch_indexes = _get_patch_indexes(max_npatches, img_npatches,
                                           batch_size, h_offset, w_offset, emb.device)

        rel_indexes, mask = _get_relative_indexes(indexes, patch_indexes,
                                                  max_npatches[0] * max_npatches[1],
                                                  support_available_out_of_range=support_out_of_range)
        # rel_indexes: (batch_size, num_embed_required)
        # mask_indexes: (batch_size, num_embed_required)

        num_embed_available = emb.shape[1]
        num_embed_required = rel_indexes.shape[1]
        if num_embed_available <= num_embed_required:
            emb = self.to_img(emb)

        emb = torch.gather(emb, dim=1, index=rel_indexes[..., None].expand(*rel_indexes.shape, emb.shape[-1]))
        emb = emb * mask[..., None]
        weights = torch.gather(weights, dim=1, index=rel_indexes) * mask
        # wieghts: (batch_size, num_embed_required)

        if num_embed_available > num_embed_required:
            emb = self.to_img(emb)
        # now emb: (batch_size, num_embed_required, patch_h * patch_w * img_channels)
        x = _unpatchify(emb, img_size, self.patch_size)
        # now x: (batch_size, height, width, img_channels)
        # we'll expand the weights to one per pixel, via repetition.
        weights = weights[..., None].expand(batch_size, num_embed_required, patch_h * patch_w)
        weights = _unpatchify(weights, img_size, self.patch_size)[..., 0]
        # now weights: (batch_size, height, width)

        return x, weights[..., None]


class MultiscalePatchExtractor(nn.Module):
    def __init__(self,
                 img_channels: int,
                 num_channels: int,
                 max_size: Tuple[int, int] = (512, 512),
                 patch_size: Tuple[int, int] = (4,4),  # must be multiple of 2.
                 num_levels: int = 3,
                 max_patch_dim_factor: int = 2):
        super().__init__()
        self.max_size = max_size
        self.patch_size = patch_size
        self.num_levels = num_levels

        def is_power_of_two(x):
            return (x & (x-1)) == 0
        assert is_power_of_two(patch_size[0]) and is_power_of_two(patch_size[1])  # power of 2
        assert max_size[0] % (patch_size[0] ** (num_levels - 1)) == 0
        assert max_size[1] % (patch_size[1] ** (num_levels - 1)) == 0

        # subsample_factors will contain how much to subsample before projecting
        # each level's patch to num_channels
        subsample_factors = [ ]
        base_patch_dim = patch_size[0] * patch_size[1] * img_channels
        # max_patch_dim is a bit arbitrary; the goal is to avoid very large projections by
        # mean-pooling over squares of 2**n pixels.
        max_patch_dim = num_channels * max_patch_dim_factor
        patch_extractors = [ ]
        patch_dims = [ ]
        index_offsets = [ 0 ]
        for level in range(num_levels):
            subsample = 1
            while base_patch_dim * (4 ** level) / (subsample ** 2) > max_patch_dim:
                subsample *= 2
            two_l = 2 ** level
            subsample_factors.append(subsample)
            patch_extractors.append(PatchExtractor(img_channels, num_channels,
                                                   (max_size[0] // subsample, max_size[1] // subsample),
                                                   (patch_size[0] * two_l // subsample, patch_size[1] * two_l // subsample)))
            # the "index_offsets" are offsets we add to the patch indexes
            # when combining multiple resolutions.
            index_offsets.append(index_offsets[-1] + patch_extractors[-1].num_indexes)

        # extractor 0 is the smallest patches.  This won't really matter.
        self.extractors = nn.ModuleList(patch_extractors)
        self.index_offsets = index_offsets
        self.subsample_factors = subsample_factors
        print("subsample = ", subsample_factors)


    def num_indexes(self):
        # the total potential number of patch indexes, over all levels.
        return self.index_offsets[-1]

    def forward(self,
                x: Tensor,
                h_offset: Optional[Tensor] = None,
                w_offset: Optional[Tensor] = None,
                num_levels: int = None) -> Tuple[Tensor, Tensor]:
        """
        Extracts all patches from a subset of levels.  (See also forward_indexes()).
        If num_levels is not supplied it defauls to len(self.extractors), i.e. to the
        num_levels given to the constructor.  If a value num_levels < self.num_levels
        is supplied, we drop the highest-resolution levels.

        Args:
          x: the image, of shape (N, H, W, C).  Its (heightsize must be a multiple of
              the largest patch size, i.e. of self.extractors[-1].patch_size
          h_offset, w_offset: these represent the offset of image x within the
              "image template" of size self.max_size; they are mostly provided
              for purposes of augmentation, but may also be useful to extract
              sub-images.
          num_levels: the number of levels of resolution; we will drop the
              higher-resolution patches if this is less than self.num_levels.


        Returns: (embeddings, indexes), where:
          embeddings: (N, num_embed, num_channels); num_embed will depend on num_levels.
             indexes: (N, num_embed),  the indexes 0 <= i < self.num_indexes()
          corresponding to the returned patches, these will depend on num_levels and
          h_offset and w_offset.
        """
        if num_levels is None:
            num_levels = self.num_levels
        assert num_levels > 0 and num_levels <= self.num_levels
        embeddings = [ ]  # one tensor per level, shape: (N, num_embeddings, num_channels)
        indexes = [ ]  # one tensor per level, shape: (N, num_embeddings)
        def _div(x, subsample):
            return x // subsample if x is not None else None
        for level in range(self.num_levels - num_levels, self.num_levels):
            s = self.subsample_factors[level]
            embed, idx = self.extractors[level](_subsample(x, s), _div(h_offset, s), _div(w_offset, s))
            embeddings.append(embed)
            indexes.append(idx + self.index_offsets[level])

        return torch.cat(embeddings, dim=1), torch.cat(indexes, dim=1)

    def forward_indexes(self,
                        x: Tensor,
                        indexes: Tensor,
                        h_offset: Optional[Tensor] = None,
                        w_offset: Optional[Tensor] = None) -> Tensor:
        """
        Version of the "forward" function that takes a list of indexes.
        Args:
               x: the image, of shape (N, C, H, W)
         indexes: the required patch indexes, of shape (N, num_embed) where
             N is batch size, with elements satisfying 0 <= i < self.num_indexes().
            All indexes must be among the set
             of indexes that would be returned by self.forward(x, h_offset, w_offset,
             self.num_levels).
        h_offset, w_offset:  these describe how the top left of the image x is offset
             from the top left of the "template image" of size self.max_size.
            num_levels: the number of resolution levels

        Returns:
             the embeddings, of shape (batch_size, num_embed, num_channels)
        """

        # emb will be a list of tensors all of shape (N, num_embed, num_channels), one
        # per level.  Each level's "emb" tensor will contain the embeddings for the
        # elements of "indexes" that correspond to that level, at the same positions.

        embs = [ ]
        def _div(x, subsample):
            return x // subsample if x is not None else None
        for level in range(self.num_levels):
            s = self.subsample_factors[level]
            embed, _indexes = self.extractors[level].forward_indexes(
                _subsample(x, s), indexes - self.index_offsets[level], _div(h_offset, s), _div(w_offset, s))
            embs.append(embed)

        index_offsets = torch.tensor(self.index_offsets, device=indexes.device)
        prev_offset = index_offsets[:-1].reshape(-1, 1, 1)
        next_offset = index_offsets[1:].reshape(-1, 1, 1)
        index_offset_mask = torch.logical_and(prev_offset <= indexes, indexes < next_offset)
        if __name__ == '__main__':
            assert torch.all((index_offset_mask.sum(dim=0) - 1.).abs() < 0.01)
        emb = torch.stack(embs, dim=0)  # embs: (num_levels, batch_size, num_embed, num_channels)
        emb = (emb * index_offset_mask.to(torch.float)[..., None]).sum(dim=0)

        return emb

    def get_child_indexes(self, indexes: Tensor) -> Tensor:
        """
        Assumes "indexes" contains only indexes of patches at the lowest resolution,
        and returns a set of indexes that contains "indexes" and also all of
        their "child indexes", meaning: all the sub-patches of these lowest-resolution
        patches.  The indexes are interspersed, so each original index has all its "child indexes"
        directly after it.  (this parent-child relationship holds at all levels).

        Args:
                indexes: (batch_size, num_indexes)
        Returns:
                augmented indexes of shape (batch_size, num_indexes2), where for example
           if self.num_levels == 3, we would have
           num_indexes2 = (1 + 4 + 16) * num_indexes = 21 * num_indexes.  The returned
           indexes could, for example, be passed to self.forward_indexes().  The order
           of the returned indexes is


        """
        (batch_size, num_indexes_in) = indexes.shape
        # all_indexes will contain all the indexes of each level, from coarse to
        # fine; don't worry about the shape, we will reshape it before combining
        # them.
        all_indexes = [ indexes ]
        # prev_indexes: (batch_size, num_indexes_in, 1); will always be: (batch_size, _, 1)
        prev_indexes = combined_indexes = indexes.unsqueeze(-1)
        prev_offset = self.index_offsets[self.num_levels - 1]
        prev_npatches = self.extractors[-1].num_patches
        for level in range(self.num_levels - 2, -1, -1):  # e.g. (1, 0) if num_levels == 3.
            this_offset = self.index_offsets[level]
            this_npatches = (prev_npatches[0] * 2, prev_npatches[1] * 2)
            # 'this_offsets' give the index offsets of the 4 patches in a 2x2 square, from the top-left
            # patch of the square.  this_npatches[1] is the width of the template image at this level;
            # it reflects that the patch_index is (x + y * width).
            this_offsets = torch.tensor([ 0, 1, this_npatches[1], this_npatches[1] + 1 ], device=indexes.device)
            # the '* 2' reflects that all the x and y patch locations are twice larger each
            # time we increase the resolution.
            this_indexes = this_offset + (prev_indexes - prev_offset) * 2  # (batch_size, _, 1)
            this_indexes = (this_indexes.unsqueeze(-1) + this_offsets) # (batch_size, _, 4)
            all_indexes.append(this_indexes)
            prev_indexes = this_indexes.reshape(batch_size, -1, 1)
            prev_offset = this_offset
            prev_npatches = this_npatches

        # intersperse the indexes.
        while len(all_indexes) > 1:
            prev = all_indexes[-2].reshape(batch_size, -1, 1)
            last = all_indexes[-1].reshape(batch_size, prev.shape[1], -1)
            all_indexes[-2] = torch.cat((prev, last), dim=2)
            all_indexes.pop()
        return all_indexes[0].reshape(batch_size, -1)


class MultiscaleImageReconstructor(nn.Module):
    def __init__(self,
                 img_channels: int,
                 num_channels: int,
                 max_size: Tuple[int, int] = (512, 512),
                 patch_size: Tuple[int, int] = (4,4),  # must be multiple of 2.
                 num_levels: int = 3,
                 max_patch_dim_factor: int = 2,
                 level_scale: float = 4.0,
                 weight_pow: float = 2.0,
                 eps: float = 0.01):
        # Note: init code is very similar to that of PatchExtractor
        super().__init__()
        self.max_size = max_size
        self.patch_size = patch_size
        self.num_levels = num_levels

        def is_power_of_two(x):
            return (x & (x-1)) == 0
        assert is_power_of_two(patch_size[0]) and is_power_of_two(patch_size[1])  # power of 2
        assert max_size[0] % (patch_size[0] ** (num_levels - 1)) == 0
        assert max_size[1] % (patch_size[1] ** (num_levels - 1)) == 0

        # subsample_factors will contain how much to subsample before projecting
        # each level's patch to num_channels
        subsample_factors = [ ]
        base_patch_dim = patch_size[0] * patch_size[1] * img_channels
        # max_patch_dim is a bit arbitrary; the goal is to avoid very large projections by
        # mean-pooling over squares of 2**n pixels.
        max_patch_dim = num_channels * max_patch_dim_factor
        reconstructors = [ ]
        patch_dims = [ ]
        index_offsets = [ 0 ]
        for level in range(num_levels):
            subsample = 1
            while base_patch_dim * (4 ** level) / (subsample ** 2) > max_patch_dim:
                subsample *= 2
            two_l = 2 ** level
            subsample_factors.append(subsample)
            reconstructors.append(ImageReconstructor(img_channels, num_channels,
                                                    (max_size[0] // subsample, max_size[1] // subsample),
                                                    (patch_size[0] * two_l // subsample, patch_size[1] * two_l // subsample)))
            # the "index_offsets" are offsets we add to the patch indexes
            # when combining multiple resolutions.
            index_offsets.append(index_offsets[-1] + reconstructors[-1].num_indexes)

        self.reconstructors = nn.ModuleList(reconstructors)
        self.index_offsets = index_offsets
        self.subsample_factors = subsample_factors
        # default_value is the value we'll use where no patch covered the image.
        self.default_value = nn.Parameter(torch.zeros(img_channels))
        # level_scale if the factor by which reconstruction from higher-resolution patches "counts more"
        # than reconstruction from lower-resolution patches.
        self.level_scale = level_scale
        # weight_pow is the power that we take the user-supplied "weights" to when reconstructing.
        # if we made this learnable it might make it too large.
        self.weight_pow = weight_pow
        # "eps" is the weight that we put on self.default_value in the weighted sum.
        #  we'll only include this weight if the remaining weights were less than
        # 0.5.
        self.eps = eps

    def num_indexes(self):
        # the total potential number of patch indexes, over all levels.
        return self.index_offsets[-1]

    def forward(self,
                emb: Tensor,
                indexes: Tensor,
                weights: Tensor,
                img_size: Tuple[int, int],
                h_offset: Optional[Tensor] = None,
                w_offset: Optional[Tensor] = None,
                num_levels: int = None) -> Tensor:
        """
        Reconstructs the image from embeddings of patches.

        If num_levels is not supplied it defauls to len(self.reconstructors), i.e. to the
        num_levels given to the constructor.  If a value num_levels < self.num_levels
        is supplied, we don't include the highest-resolution levels.  num_levels
        must correspond to the num_levels given to MultiscalePatchExtractor.

        Args:
               emb:  (N, num_embed, C_emb), the input embeddings; N is batch size.
           weights: (N, num_embed), the weights corresponding to each of
                these embeddings, to ensure differentiablility, as returned by
                class Subset()'s forward function.  Implicitly, non-present
                embeddings have weight zero.  The weights will be processed
                and returned at the pixel level.
             img_size: the size of the image x that we are trying to reconstruct,
                in pixels
           h_offset, w_offset:  offsets in pixels into the template
               image of size self.max_size.  Must be a multiple of
               the patch size.
          num_levels: the number of levels of resolution to process; we will drop the
              higher-resolution patches if this is less than self.num_levels.

        Returns:  the reconstructed image, of shape (N, H, W, C).
        """
        if num_levels is None:
            num_levels = self.num_levels
        assert num_levels > 0 and num_levels <= self.num_levels
        images = [ ]  # one tensor per level, shape: (N, H, W, C)
        img_weights = [ ]  # one tensor per level, shape: (N, H, W)
        def _div(x, subsample):
            return x // subsample if x is not None else None
        for level in range(self.num_levels - num_levels, self.num_levels):
            s = self.subsample_factors[level]
            _img_size = (img_size[0] // s, img_size[1] // s)
            this_indexes = indexes - self.index_offsets[level]
            img, weight = self.reconstructors[level](emb, this_indexes, weights, _img_size,
                                                     _div(h_offset, s), _div(w_offset, s))
            images.append(_upsample(img, s))
            weight = (weight ** self.weight_pow) * (self.level_scale ** -level)
            img_weights.append(_upsample(weight, s))

        num_channels = self.default_value.numel()
        images.append(self.default_value.reshape(1, 1, 1, num_channels).expand_as(images[-1]))
        img_weights.append(torch.full((1, 1, 1, 1), self.eps, device=emb.device).expand_as(img_weights[-1]))

        print("mean of img_weights is: ", [ x.mean().item() for x in img_weights ] )
        print("std of img is: ", [ x.std().item() for x in images ] )

        weighted_img = torch.stack([ i * w for i, w in zip(images, img_weights) ], dim=0).sum(dim=0)
        img_weight = torch.stack(img_weights, dim=0).sum(dim=0)
        img = weighted_img / img_weight
        print("std of final img is: ", img.std())
        return img


def _test_sort_numbers():
    num_channels = 128
    model = Subset(num_channels=num_channels, noise_bias=False)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=2e-04)

    for step in range(10000):
        batch_size = 10
        num_embeddings = 256
        N = 32  # keep N embeddings

        x = torch.randn(batch_size, num_embeddings, num_channels)

        _indexes, y = model(x, N)

        # want the N largest elements of the 1st dim of x
        loss = - y[..., 0].sum()

        # ref_loss is the best we could have done.

        values, _indexes2 = x[..., 0].sort(dim=-1)
        ref_loss = -values[:,-N:].sum()

        if step % 100 == 0:
            den = N * batch_size

            # get loss in eval mode
            model.eval()
            _indexes, y = model(x, N)
            model.train()
            # want the N largest elements of the 1st dim of x
            loss_eval = - y[..., 0].sum()
            print(f"step={step}, loss={loss.item()/den}, loss_eval={loss_eval.item()/den}, ref_loss={ref_loss.item()/den}, to_scores={model.to_scores.weight}")

        loss.backward()
        optim.step()
        optim.zero_grad()



def _test_patchify():
    x = torch.randn(10, 64, 32, 3)
    patch_size = (4, 8)
    emb = _patchify(x, patch_size)
    x2 = _unpatchify(emb, (x.shape[1], x.shape[2]), patch_size)
    assert torch.allclose(x, x2)


def _test_get_patch_indexes():
    if True:
        y = _get_patch_indexes((2, 4), (2, 4), 1, None, None, torch.device('cpu'))
        print("y = ", y)
        assert torch.all(y == torch.tensor([[ 0, 1, 2, 3, 4, 5, 6, 7 ]]))

    if True:
        y = _get_patch_indexes((2, 4), (2, 2), 1, None, None, torch.device('cpu'))
        print("y = ", y)
        assert torch.all(y == torch.tensor([[ 0, 1,  4, 5 ]]))

    if True:
        y = _get_patch_indexes((2, 4), (2, 2), 2, torch.tensor([0, 0]), torch.tensor([0, 1]), torch.device('cpu'))
        print("y = ", y)
        assert torch.all(y == torch.tensor([ [ 0, 1,  4, 5 ],
                                             [ 1, 2, 5, 6 ]]))


def _test_get_relative_indexes():
    if True:
        available = torch.tensor( [ [ 0, 1, 2, 3 ],
                                    [ 4, 5, 6, 7 ] ] )
        required = torch.tensor( [  [ 0, 2, 4, 6, 8 ],
                                    [ 2, 4, 6, 8, 10] ])
        indexes = torch.tensor( [ [ 0, 2, -1, -1, -1 ],
                                  [ -1, 0, 2, -1, -1 ] ])

        for required_size in [ 1, 2, 3, 4, 5 ]:
            ans, mask = _get_relative_indexes(available, required[:,:required_size], N=8,
                                              support_required_out_of_range=True)
            # TODO: should test support_required_out_of_range
            idxs = indexes[:, :required_size]
            assert(torch.all(mask.to(torch.bool) == (idxs != -1)))
            assert(torch.all(torch.where(mask.to(torch.bool), ans, idxs) == idxs))


def _init_default(m):  # used in testing.
    with torch.no_grad():
        if isinstance(m, PatchExtractor):
            m.to_embed.bias.zero_()
            m.to_embed.weight.zero_()
            n = min(m.to_embed.weight.shape)
            m.to_embed.weight[:n, :n] = torch.eye(n)
        elif isinstance(m, ImageReconstructor):
            m.to_img.bias.zero_()
            m.to_img.weight.zero_()
            n = min(m.to_img.weight.shape)
            m.to_img.weight[:n, :n] = torch.eye(n)
        elif isinstance(m, MultiscalePatchExtractor):
            for n in m.extractors:
                _init_default(n)
        elif isinstance(m, MultiscaleImageReconstructor):
            for n in m.reconstructors:
                _init_default(n)


def _test_patch_extractor_reconstructor():
    img_channels = 3
    num_channels = 256
    max_size = (512, 512)
    patch_size = (4, 4)

    m = PatchExtractor(img_channels, num_channels, max_size, patch_size)
    r = ImageReconstructor(img_channels, num_channels, max_size, patch_size)
    _init_default(m)
    _init_default(r)

    batch_size = 4
    img_size = (64, 32)
    x = torch.randn(batch_size, *img_size, img_channels)
    y, indexes = m(x)
    assert y.shape == (batch_size, (img_size[0]*img_size[1])//(patch_size[0]*patch_size[1]),
                       num_channels)
    print("patch_extractor: indexes = ", indexes)

    h_offset = patch_size[0] * torch.tensor([ 0, 7, 5, 3 ] )
    w_offset = patch_size[1] * torch.tensor([ 10, 11, 9, 0 ] )

    y, indexes = m(x, h_offset, w_offset)
    weights = torch.ones(*indexes.shape)
    assert y.shape == (batch_size, (img_size[0]*img_size[1])//(patch_size[0]*patch_size[1]),
                       num_channels)
    print("patch_extractor[offset]: indexes = ", indexes)

    for support_out_of_range in [False, True]:
        # nothing is out of range so should be OK in either case.
        x_recon, weights2 = r(y, indexes, weights, img_size, h_offset, w_offset, support_out_of_range)
        assert torch.allclose(x, x_recon)



def _test_multiscale():
    img_channels = 3
    num_channels = 256
    max_size = (512, 512)
    patch_size = (4, 4)
    num_levels = 3

    m = MultiscalePatchExtractor(img_channels, num_channels, max_size, patch_size,
                                 num_levels)
    _init_default(m)
    r = MultiscaleImageReconstructor(img_channels, num_channels, max_size, patch_size)
    _init_default(r)

    batch_size = 4
    img_size = (64, 32)
    x = torch.randn(batch_size, *img_size, img_channels)
    y, indexes = m(x)

    print("patch_extractor: indexes = ", indexes)

    h_offset = patch_size[0] * torch.tensor([ 0, 7, 5, 3 ] )
    w_offset = patch_size[1] * torch.tensor([ 10, 11, 9, 0 ] )

    y, indexes = m(x, h_offset, w_offset)
    weights = torch.ones(*indexes.shape)

    print("patch_extractor[offset]: indexes = ", indexes)

    # nothing is out of range so should be OK in either case.
    x_recon = r(y, indexes, weights, img_size, h_offset, w_offset)
    # it's inexact for a few reasons, including the "backoff" mechanism for unseen patches
    # and loss of resolution in the lowest-resolution patches.
    cosine = (x_recon * x).sum() / ((x_recon ** 2).sum() * (x ** 2).sum()).sqrt()
    print("cosine distance between x and x_recon is ", cosine)
    assert cosine > 0.99


def _test_get_child_indexes():
    img_channels = 3
    num_channels = 256
    max_size = (512, 512)
    patch_size = (4, 4)
    num_levels = 2

    m = MultiscalePatchExtractor(img_channels, num_channels, max_size, patch_size,
                                 num_levels)
    _init_default(m)
    r = MultiscaleImageReconstructor(img_channels, num_channels, max_size, patch_size)
    _init_default(r)

    batch_size = 4
    img_size = (64, 32)
    x = torch.randn(batch_size, *img_size, img_channels)

    h_offset = patch_size[0] * torch.tensor([ 0, 7, 5, 3 ] )
    w_offset = patch_size[1] * torch.tensor([ 10, 11, 9, 0 ] )

    y_coarse, indexes_coarse = m(x, h_offset, w_offset, num_levels=1)  # only largest patches.

    indexes = m.get_child_indexes(indexes_coarse)

    y = m.forward_indexes(x, indexes, h_offset, w_offset)

    weights = torch.ones(*indexes.shape)

    print("get_child_indexes, patch_extractor[offset]: indexes = ", indexes.sort(dim=-1)[0])
    if True:
        y_all, indexes_all = m(x, h_offset, w_offset)
        print("get_child_indexes, all indexes = ", indexes_all.sort(dim=-1)[0])

    # nothing is out of range so should be OK in either case.
    x_recon = r(y, indexes, weights, img_size, h_offset, w_offset)
    # it's inexact for a few reasons, including the "backoff" mechanism for unseen patches
    # and loss of resolution in the lowest-resolution patches.
    cosine = (x_recon * x).sum() / ((x_recon ** 2).sum() * (x ** 2).sum()).sqrt()
    print("cosine distance between x and x_recon is ", cosine)
    assert cosine > 0.99





if __name__ == '__main__':
    _test_patchify()
    _test_get_patch_indexes()
    _test_get_relative_indexes()
    _test_patch_extractor_reconstructor()
    _test_multiscale()
    _test_get_child_indexes()



"""
    Zengwei, I have something I'd like you try for me related to image recognition.  See the attached ranking.py, which will be needed.  This thing is on quite
high priority, higher than the stu

IDEA:
  The basic idea here is to use transformers for image recognition, but both in training and inference,
  to use a subset of the patches in the image; and also to introduce patches with multiple resolutions
  (so the model can "back off" to lower resolutions).

  The overall structure will be like MAE transformer, but with a couple extra things and some changes.

  The baseline is:

   embed pixels per patch -> [transformer]  -> patch embeddings

  Then the patch embeddings are used as auxiliary input "src" of a transformer decoder,
  with the pos-emb's of a target patch as the main input.




"""
