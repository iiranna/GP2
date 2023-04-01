import troch 
import troch.nn as nn

  class PatchEmbed(nn.Module):
    """Parameter
    img_size=int
    patch_size=int
    in_chans=int
    embed_dim=int
    Attributes
    n_patches=int
    proj:nn.Conv2d"""

    def__init__(selef, img_size, patch_size, in_chans=3, embed_dim=768):
    super().__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_patches = (img_size // patch_size) ** 2

    self.proj = nn.Conv2d(
        in_chans,
        embed_dim,
        kernal_size = patch_size,
        stride = patch_size
    )

    def forward(self,x) :
      """Run forward pass.
      Parameters: x:torch.Tensor
        shape (n_samples, in_chans, img_size, patch_size).
      Returns: torch.Tensor
        shape (n_samples, n_patches, embed_dim)."""

      x = self.proj(
          x
      )
      x = x.flatten(2)
      x = x.transpose(1,2)

      return x

      class Attention(nn.Module):
        """Attention mechanism.
        Parameters: 
        dim=int
        n_heads=int
        qkv_bias=bool
        attn_p=float
        proj_p=float"""

      def__init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super(). __init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)