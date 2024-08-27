import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Optional
import random
from patches import MultiscalePatchExtractor, MultiscaleImageReconstructor, _init_default
from ranking import SubsetItems
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

try:
    from scaling import FloatLike
except:
    FloatLike = float



class SubsetPatchEmbeddings(nn.Module):
    def __init__(self,
                 num_channels: int,
                 num_levels: int,
                 noised_fraction: float = 0.1,
                 stddev_factor: float = 4.0,
                 use_random_rank: bool = True,
                 random_fraction: FloatLike = 0.1):
        """
        Class that helps to compute subsets of patches in an image task.

        Args:
               num_channels:  the number of channels in the embeddings
                 num_levels: the number of levels of pathes, as in MultiscalePatchExtractor
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

        # e.g. 1 + 4 + 16 = 21.
        patches_per_toplevel_patch = sum( [ 4 ** n for n in range(num_levels) ] )

        self.emb_scale = nn.Parameter(torch.ones(num_channels))

        self.to_scores = nn.Linear(num_channels, patches_per_toplevel_patch)
        self.subset_items = SubsetItems(noised_fraction=noised_fraction,
                                        stddev_factor=stddev_factor,
                                        use_random_rank=use_random_rank,
                                        random_fraction=random_fraction)

    def forward(self,
                img: Optional[Tensor],
                emb: Tensor,
                indexes_toplevel: Tensor,
                m: MultiscalePatchExtractor,
                pos_emb: nn.Embedding,
                h_offset: Optional[Tensor],
                w_offset: Optional[Tensor],
                N: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward function that selects some elements of the embeddings x and returns embeddings,
        but without noise added yet; call add_noise() later for that.

.  Args:
        img: the original input image, including mask; of shape (batch_size, height, width, img_channels + 1);
            or None if you don't want to include the image in the embedding.
        emb: the embeddings to select from, one per top-level patch, of shape (batch_size, num_embeddings, num_channels))
     indexes_toplevel: the patch indexes corresponding to the top-level patches
          m: object of type MultiscalePatchExtractor to compute embeddings for chosen patches.
    pos_emb: object to compute positional embeddings
          N: the number of embeddings from 'num_embeddings' to keep.
  h_offset,w_offset: (batch_size,), offsets of each image within the template image.

        Returns: (y, index, weight), where:
                 y: (batch_size, N, num_channels), the selected features, these will include contributions
                       from the embeddings of the top-level patches; from the image pixels; from the
                       positional embeddings; and a noise term for the almost-not-selected patches.
             index: (batch_size, N), the indexes of the chosen embeddings
            weight: (batch_size, N), weights between 0 and 1 that will
                    only be != 1 for a fraction "noised_fraction" of the weights.
        """
        scores = self.to_scores(emb)  # (batch_size, num_emb, patches_per_toplevel_patch)
        (batch_size, num_emb, patches_per_toplevel_patch) = scores.shape
        scores = scores.reshape(batch_size, num_emb * patches_per_toplevel_patch)
        emb_indexes = (torch.arange(num_emb * patches_per_toplevel_patch, device=emb.device) //
                       patches_per_toplevel_patch)  # indexes into 'x' that give us the top-level patch
        emb_indexes = emb_indexes.unsqueeze(0).expand(batch_size, num_emb * patches_per_toplevel_patch)

        # patch_indexes: (batch_size, num_emb * patches_per_toplevel_patch)
        patch_indexes = m.get_child_indexes(indexes_toplevel)

        index, weight = self.subset_items(scores, N)
        # index, weight: each (batch_size, N)

        # gather the chosen subsets.
        emb_indexes = torch.gather(emb_indexes, dim=1, index=index)
        # now emb_indexes: (batch_size, N)
        patch_indexes = torch.gather(patch_indexes, dim=1, index=index)
        # now patch_indexes: (batch_size, N)

        # the following looks up the contribution to the embedding from the "frontend embedding".  for now
        # we use the same scales for all patch levels.
        x = torch.gather(emb * self.emb_scale, dim=1, index=emb_indexes[..., None].expand(batch_size, N, emb.shape[-1]))

        x = x + pos_emb(patch_indexes)

        if img is not None:
            x = x + m.forward_indexes(img, patch_indexes, h_offset, w_offset)

        x = self.subset_items.add_noise(x, weight)
        return x, index, weight



class MultiscaleVisionModel(nn.Module):
    # this one is to be trained both using MAE and also classification.
    def __init__(self,
                 img_channels: int,  # e.g. 3
                 num_channels: int,
                 num_levels: int, # e.g. 3, for patches.
                 patch_size: int, # e.g. (4, 4)
                 max_size: Tuple[int, int],  # max image size, e.g. (512, 512)
                 frontend_layers: int,
                 encoder_layers: int,
                 decoder_layers: int,
                 num_classes: int):
        super().__init__()
        # we add a channel to the image before giving it to the extractor, to be
        # the mask reflecting what part of the image, if any, is masked and we
        # are going to be doing MAE loss on that part.
        self.extractor = MultiscalePatchExtractor(img_channels + 1,
                                                  num_channels,
                                                  max_size,
                                                  patch_size,
                                                  num_levels)

        self.reconstructor_frontend = MultiscaleImageReconstructor(img_channels,
                                                                   num_channels,
                                                                   max_size,
                                                                   patch_size,
                                                                   num_levels)

        self.reconstructor_decoder = MultiscaleImageReconstructor(img_channels,
                                                                  num_channels,
                                                                  max_size,
                                                                  patch_size,
                                                                  num_levels)


        encoder_layer = TransformerEncoderLayer(d_model=num_channels,
                                                nhead=8,
                                                dim_feedforward=4 * num_channels,
                                                dropout=0.1,
                                                activation="gelu",
                                                norm_first=True,
                                                batch_first=True)


        self.frontend = TransformerEncoder(encoder_layer=encoder_layer,
                                           num_layers=frontend_layers)

        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=encoder_layers)

        # the decoder is for the MAE reconstruction.
        # For classification, we
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model=num_channels,
                                    nhead=8,
                                    dim_feedforward=4 * num_channels,
                                    dropout=0.1,
                                    activation="gelu",
                                    norm_first=True,
                                    batch_first=True),
            num_layers=decoder_layers)


        # the following objects compute the patch features for the sets of
        # patches used in the encoder and decoder respectively.
        self.encoder_subset = SubsetPatchEmbeddings(num_channels, num_levels)
        self.decoder_subset = SubsetPatchEmbeddings(num_channels, num_levels)


        self.pos_emb = nn.Embedding(self.extractor.num_indexes(),
                                    num_channels)

        self.head_frontend = nn.Linear(num_channels, num_classes)
        self.head_encoder = nn.Linear(num_channels, num_classes)

        self.loss_frontend = nn.CrossEntropyLoss()  # mean reduction is default
        self.loss_encoder = nn.CrossEntropyLoss()


    def forward(self,
                x: Tensor,
                mask: Tensor,
                y: Tensor,
                num_encoder_patches: int,
                num_decoder_patches: int,
                h_offset: Optional[Tensor],
                w_offset: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward function that computes classification and MAE losses.
        Args:
          x: the input image, of shape (N, H, W, C), e.g. C == 3
       mask: (N, H, W), the mask for the input image that's 1 where we want to mask the image
          and compute the MAE loss, and 0 elsewhere.
          y: (N,), containing indexes 0 <= i < num_classes, the class labels for purpose of
             computing classification loss.
      h_offset, w_offset:  offsets in pixels from the top left of the image template of
            shape self.max_size, provided mostly for augmentation.

     encoder_patches: the number of patches to select for use in the encoder
     decoder_patches: the number of patches to select for use in the decoder

        Returns:
          (loss_frontend_classification,
           loss_encoder_classification,
           loss_frontend_mae,
           loss_decoder_mae)
        which are all scalar Tensors.
        """
        img_shape = (x.shape[1], x.shape[2])
        x_masked = x * (1.0 - mask.unsqueeze(-1))
        x_and_mask = torch.cat((x_masked, mask.unsqueeze(-1)), dim=-1)  # e.g. 4 channels now

        input_emb, toplevel_indexes = self.extractor(x_and_mask, h_offset, w_offset, num_levels=1)
        # input_emb: (batch_size, num_patches, num_channels)

        input_emb = input_emb + self.pos_emb(toplevel_indexes)

        frontend_emb = self.frontend(input_emb)
        # frontend_emb: (batch_size, num_patches, num_channels)

        encoder_input_emb, encoder_index, _encoder_weights = self.encoder_subset(
            x_and_mask, frontend_emb, toplevel_indexes, self.extractor,
            self.pos_emb, h_offset, w_offset, num_encoder_patches)

        # the None is where the image would be; don't include the image pixels
        # in the decoder input, it has to get everything from the encoder's output
        # and from the frontend.
        decoder_input_emb, decoder_index, decoder_weights = self.decoder_subset(
            None, frontend_emb, toplevel_indexes, self.extractor,
            self.pos_emb, h_offset, w_offset, num_decoder_patches)

        encoder_output_emb = self.encoder(encoder_input_emb)

        decoder_output_emb = self.decoder(decoder_input_emb,
                                          encoder_output_emb)

        # reconstruct the image from the decoder output.
        # _img_decoder_weights is the same shape as `img`, it is
        # something you may want to visualize to see where the decoder
        # patches are.
        img_recon_decoder, _img_decoder_weights = self.reconstructor_decoder(decoder_output_emb,
                                                                             decoder_index,
                                                                             decoder_weights,
                                                                             img_shape, h_offset, w_offset)

        if True:
            # The following things are not needed for computing the losses; in
            # fact _img_recon_encoder will contain nonsense as this path is
            # never used in any loss.  The point is to show how to visualize
            # where the encoder patches are; you can visualize
            # _img_encoder_weights to see where they are.  You may want to
            # scale it by dividing by the max.
            _img_recon_encoder, _img_encoder_weights = self.reconstructor_decoder(encoder_output_emb,
                                                                                  encoder_index,
                                                                                  _encoder_weights,
                                                                                  img_shape, h_offset,
                                                                                  w_offset)

            if True:
                # Caution: when we visualize the patches below, we dont expect to
                # see all the patches inside the visualized portion until the model is
                # trained, because actually the chosen patches can be anywhere in the
                print("encoder_weights = ", _img_encoder_weights)
                enc_patches = _img_encoder_weights / _img_encoder_weights.max()
                dec_patches = _img_decoder_weights / _img_decoder_weights.max()
                decoder_img = (img_recon_decoder + 1) * 0.5  # not really sure about this formula
                input_img = (x + 1) * 0.5
                import matplotlib.pyplot as plt
                img = torch.cat((mask[..., None].expand_as(x), enc_patches, dec_patches, decoder_img, input_img), dim=2)
                plt.imshow(img.detach()[0], cmap='hot', interpolation='nearest')
                plt.show()



        # reconstruct the image from the frontend output.
        # (The patch locations in this case are not interesting so we name the 2nd output as
        # _ and we'll discard it.)
        img_recon_frontend, _ = self.reconstructor_frontend(frontend_emb,
                                                            toplevel_indexes,
                                                            torch.ones(*toplevel_indexes.shape, device=x.device),
                                                            img_shape, h_offset, w_offset)

        mae_loss_frontend = (((img_recon_frontend - x) ** 2) * mask[..., None]).mean()
        mae_loss_decoder = (((img_recon_decoder - x) ** 2) * mask[..., None]).mean()

        classification_loss_frontend = self.loss_frontend(
            self.head_frontend(frontend_emb.mean(dim=1)),
            y)
        classification_loss_encoder = self.loss_encoder(
            self.head_encoder(encoder_output_emb.mean(dim=1)),
            y)

        return (classification_loss_frontend,
                classification_loss_encoder,
                mae_loss_frontend,
                mae_loss_decoder)



# return's a mask that's 1 in a randomly located square of size 'mask_size',
# and zero elsewhere.
def get_rand_mask(batch_size: int,
                  img_size: Tuple[int, int],
                  mask_size: Tuple[int, int],
                  num_patches: int,
                  device):
    ans = None
    arange_h = torch.arange(img_size[0], device=device).unsqueeze(-1)
    arange_w = torch.arange(img_size[1], device=device)
    for s in range(num_patches):
        mask_h_offset = torch.randint(img_size[0] - mask_size[0], size=(batch_size,), device=device)[:, None, None]
        mask_w_offset = torch.randint(img_size[1] - mask_size[1], size=(batch_size,), device=device)[:, None, None]

        mask = torch.logical_and(
            torch.logical_and(arange_h >= mask_h_offset, arange_h < mask_h_offset + mask_size[0]),
            torch.logical_and(arange_w >= mask_w_offset, arange_w < mask_w_offset + mask_size[1]))
        if s == 0:
            ans = mask
        else:
            ans = torch.logical_or(mask, ans)
    assert ans.shape == (batch_size, *img_size)
    return ans.to(torch.float)



def _test_get_rand_mask():
    num_patches = 2
    rand_mask = get_rand_mask(2, (10, 16), (2, 5), num_patches, 'cpu')
    print("mask = ", rand_mask)

def _test_multiscale():
    img_channels = 3
    num_channels = 256  # might use 512 in reality.
    max_size = (512, 512)  # the size of the "template image", which will dictate how large images we can
                           # process after the model is trained.
    patch_size = (4, 4)  # In reality we'd probably have the patches a bit larger, like (8, 8).
                         # This is the "smallest patch size" of 3 levels of patches, so the
                         # largest patch would be 32 x 32.  We can make the images larger
                         # than normal, i.e. process at higher resolution.

    img_size = (64, 64)  # the size of the images we train on.  Actually we could use a much
                         # larger image size than normal; probably at least 256 x 256 or even
                         # more.  Initially, try double the width and height of whatever you'd
                         # normally be using.

    num_levels = 3
    num_classes = 10

    model = MultiscaleVisionModel(img_channels, num_channels, num_levels,
                                  patch_size, max_size,
                                  frontend_layers=3, # would use more
                                  encoder_layers=3,  # prob. the encoder would have the most layers.
                                  decoder_layers=3,
                                  num_classes=num_classes)

    model.train()

    batch_size = 10
    img = torch.randn(batch_size, *img_size, 3)
    lps = model.extractor.largest_patch_size
    # choose random height and width offsets
    h_offset = lps[0] * torch.randint((max_size[0] - img_size[0]) // lps[0], size=(batch_size,))
    w_offset = lps[1] * torch.randint((max_size[1] - img_size[1]) // lps[1], size=(batch_size,))

    mask_size = (32, 32) # mask and predict square patches of this size in the
                         # image, of number `num_patches`; the rest is visible.

    num_patches = 2
    mask = get_rand_mask(batch_size, img_size, mask_size, num_patches, 'cpu')


    #optim = torch.optim.Adam(model.parameters(), lr=2e-04)

    y = torch.randint(num_classes, size=(batch_size,))

    _init_default(model)

    # we would most likely have more encoder and decoder patches than just 64.
    # e.g. could use 128 or 256 patches.
    losses = model(img, mask, y, num_encoder_patches=64, num_decoder_patches=64,
                   h_offset=h_offset, w_offset=w_offset)

    print("losses = ", losses)



if __name__ == '__main__':
    _test_get_rand_mask()
    _test_multiscale()
