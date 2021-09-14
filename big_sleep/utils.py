##Some utility functions
import torch
import torch.nn
import torch.nn.functional as F

from typing import List, Optional

#For now, just replicate PyTorch's _in_projection_packed. Sorry for stealing code (again!)
def in_proj_packed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, b: Optional[Tensor] = None,) -> List[Tensor]:
	"""A reimplementation of PyTorch's _in_projection_packed function to allow for transferring weights"""