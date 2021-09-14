import math

from typing import Tuple

import torch
import torch.nn.functional as F

from entmax import entmax15
from torch import nn

from .utils import exists

#reimplementing MultiheadAttention in a manner very similar to torch.nn.functional and torch.nn.MultiheadAttention()'s source code as an experiment.
#Would this allow for a modified architecture to be used...?
class MultiheadAttentionMod(nn.Module):
	def __init__(self, embed_dim, num_heads, dropout=0.):
		super(MultiheadAttentionMod, self).__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.dropout = dropout
		self.head_dim = embed_dim // num_heads

		#We assume since this is used for CLIP only that q, k, and v have same embed dim no matter what. No checks
		self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
		#and that bias is present as well, no matter what
		self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

		nn.init.xavier_normal_(self.in_proj_weight)
		nn.init.constant_(self.in_proj_bias, 0.)
		nn.init.constant_(self.out_proj.bias, 0.)

	def in_proj_packed(self, query, key, value, weight, bias=None):
		#reimplementation of _in_projection_packed from pytorch source code
		query = query.half()
		key = key.half()
		value = value.half()
		weight = weight.half()
		if exists(bias):
			bias = bias.half()
		embed_dim = query.size(-1)

		if key is value:
			if query is key:
				#self attention
				return F.linear(query, weight, bias).chunk(3, dim=-1)
			else:
				#encoder-decoder attention
				weight_q, weight_kv = weight.split([embed_dim, embed_dim * 2])
				if bias is None:
					bias_q = bias_kv = None
				else:
					bias_q = bias_kv = bias.split([embed_dim, embed_dim * 2])

		else:
			weight_q, weight_k, weight_v = weight.chunk(3)
			if bias is None:
				bias_q = bias_k = bias_v = None
			else:
				bias_q, bias_k, bias_v = bias.chunk(3)

			return F.linear(query, weight_q, bias_q), F.linear(key, weight_k, bias_k), F.linear(value, weight_v, bias_v)

	def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
		#scaled dot product attention. 'nuff said
		batch_size, source_len, embed_dim = query.shape
		query = query / math.sqrt(embed_dim)
		#if i can figure out einops i'd do some changing here
		# (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
		attn = torch.bmm(query, key.transpose(-2, -1))

		if exists(attn_mask):
			attn += attn_mask

		#softmax... replaced with entmax soon
		attn = F.softmax(attn, dim=-1)
		if dropout > 0.0:
			attn = F.dropout(attn, p=dropout)
		# (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
		output = torch.bmm(attn, value)

		return output, attn

	def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True):
		target_length, bsz, embed_dim = query.shape
		source_length, _, _ = key.shape
		if isinstance(embed_dim, torch.Tensor):
			#embed_dim can be a tensor when JIT tracing
			head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
		else:
			head_dim = embed_dim // self.num_heads

		assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

		#compute in-projection
		q, k, v = self.in_proj_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

		#just uh copy this code bit here (pls no sue facebook thx)
		# prep attention mask
		if exists(attn_mask):
			if attn_mask.dtype == torch.uint8:
				#warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
				attn_mask = attn_mask.to(torch.bool)
			else:
				assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
				f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
			# ensure attn_mask's dim is 3
			if attn_mask.dim() == 2:
				correct_2d_size = (target_length, source_length)
				if attn_mask.shape != correct_2d_size:
					raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
				attn_mask = attn_mask.unsqueeze(0)
			elif attn_mask.dim() == 3:
				correct_3d_size = (bsz * num_heads, target_length, source_length)
				if attn_mask.shape != correct_3d_size:
					raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
			else:
				raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

		# prep key padding mask
		if exists(key_padding_mask) and key_padding_mask.dtype == torch.uint8:
			#warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
			key_padding_mask = key_padding_mask.to(torch.bool)

	     #reshape q, k, v for multihead attention and make 'em batch first
		q = q.contiguous().view(target_length, bsz * self.num_heads, head_dim).transpose(0, 1)
		k = k.contiguous().view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
		v = v.contiguous().view(v.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)

		source_length = k.size(1)

		#merge key padding and attention masks
		if exists(key_padding_mask):
			assert key_padding_mask.shape == (bsz, src_len), \
				f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
			key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
				expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
			if attn_mask is None:
				attn_mask = key_padding_mask
			elif attn_mask.dtype == torch.bool:
				attn_mask = attn_mask.logical_or(key_padding_mask)
			else:
				attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

		#convert the mask to float if bool
		if exists(attn_mask) and attn_mask.dtype == torch.bool:
			new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
			new_attn_mask.masked_fill_(attn_mask, float("-inf"))
			attn_mask = new_attn_mask

		#(deep breath [copier's note: for me too]) calculate attention and out projection
		attn_output, attn_output_weights = self.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout)
		attn_output = attn_output.transpose(0, 1).contiguous().view(target_length, bsz, embed_dim)
		attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

		if need_weights:
			#average attention weights over heads
			attn_output_weights = attn_output_weights.view(bsz, self.num_heads, target_length, source_length)
			return attn_output, attn_output_weights.sum(dim=1) / num_heads
		else:
			return attn_output, None