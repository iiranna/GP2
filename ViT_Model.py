import troch 
import troch.nn as nn

  class PatchEmbed(nn.Module):
    """Parameter
    img_size = int
    patch_size = int
    in_chans = int >> values for all the three channels (r,g,b) of every pixels
    embed_dim = int >> number of channels * patche size = 16*16*3
    
    Attributes
    n_patches = int
    proj:nn.Conv2d"""

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
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
      """ Run forward pass.
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
        """ Attention mechanism.
         Parameters: 
          dim=int
          n_heads=int
          qkv_bias=bool
          attn_p=float
          proj_p=float """

      def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
        def forward(self,x):
          """Run forward pass.
          Parameters: 
          x : torch_Tensor
            shape (n_samples, n_patches +1, dim).
          Returns: 
          torch_tensor
            shape (n_samples, n_patches +1, dim)."""
            
          n_samples, n_tokens, dim = x.shape
          if dim != self.dim
          raise ValueError
          
          qkv = self.qkv(x)
          qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim)
          qkv = qkv.permute(2, 0, 3, 1, 4)
          q,k,v = qkv[0], qkv[1], qkv[2]
          k_t = k.transpose(-2, -1)
          dp = (q @ k_t) * self.scale
          attn = dp.softmax(dim = -1)
          attn = self.attn_drop(attn)
          weighted_avg = attn @ v
          weighted_avg = weighted_avg.transpose(1, 2)
          weighted_avg = weighted_avg.flatten(2)
          
          x = self.proj(weighted_avg)
          x = self.proj_drop(x)
            return x

  class MLP(nn.Module):
      """Multilayer perceptron"""
      def __init__(self, in_features, hidden_features, out_features, p=0.):
      super().__init()
      self.fc1 = nn.Linear(in_features, hidden_features)
      self.act = nn.GELU()
      self.fc2 = nn.Linear(hidden_features, out_features)
      self.drop = nn.Dropout(p)

      def forward(self, x):

      x = self.fc1(
              x
      )
      x = self.act(x) 
      x = self.drop(x)
      x = self.fc2(x)
      x = self.drop(x)

      return x
 
 class Block(nn.Module):
    """Transformer block"""
    
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
      super().__init__()
      self.norm1 = nn.LayerNorm(dim, eps=1e-6)
      self.attn = Attention(
              dim,
              n_heads = n_heads,
              qkv_bias = qkv_bias,
              attn_p = attn_p,
              proj_p = p
      )
      
      self.norm2 = nn.LayerNorm(dim, eps=1e-6)
      hidden_features = int(dim * mlp_ratio)
      self.mlp = MLP(
              in_fearures = dim,
              hidden_features = hidden_features,
              out_features = dim,
      )
      
      def forward(self, x):
      
      x = x + self.attn(self.norm1(x))
      x = x + self.mlp(self.norm2(x))
      
      return x
