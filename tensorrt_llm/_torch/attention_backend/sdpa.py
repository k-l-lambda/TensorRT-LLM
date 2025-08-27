
import torch



def vanilla (q, k, v, attn_mask):
	print(f'{q.shape=}, {k.shape=}, {v.shape=}, {attn_mask.shape=}')
	head_dim = q.size(1)
	s = torch.einsum('bhsd,bhSd->bhsS', q, k)

	s /= head_dim**0.5

	s = s.masked_fill(attn_mask == 0, float('-inf'))
	s = torch.softmax(s, dim=-1)

	out = torch.matmul(s, v)

	return out


def sparse (q, k, v, attn_mask):
	batch, num_heads, q_len, kv_len = s.shape
	s = torch.einsum('bhsd,bhSd->bhsS', q, k)

	s /= num_heads**0.5

	# Parameters for the mask (set these as needed)
	block_size = 16
	num_local_blocks = 2
	vertical_stride = 32
	homo_head_pattern = True  # or False, depending on your use case

	device = s.device

	# Compute row and col indices
	row_idx = torch.arange(q_len, device=device).view(-1, 1).expand(q_len, kv_len)
	col_idx = torch.arange(kv_len, device=device).view(1, -1).expand(q_len, kv_len)

	block_row_idx = row_idx // block_size
	block_col_idx = col_idx // block_size

	block_local_mask = (block_row_idx - block_col_idx) < num_local_blocks

	if homo_head_pattern:
		head_sliding_step = 0
	else:
		head_sliding_step = max(1, vertical_stride // num_heads)

	# Prepare head_idx for broadcasting
	head_idx = torch.arange(num_heads, device=device).view(-1, 1, 1)
	block_col_idx_b = block_col_idx.unsqueeze(0).expand(num_heads, q_len, kv_len)
	block_vertical_stride_mask = ((block_col_idx_b + head_idx * head_sliding_step + 1) % vertical_stride) == 0

	causal_mask = (col_idx <= row_idx).unsqueeze(0).expand(num_heads, q_len, kv_len)
	block_local_mask_b = block_local_mask.unsqueeze(0).expand(num_heads, q_len, kv_len)

	combined_mask = causal_mask & (block_local_mask_b | block_vertical_stride_mask)

	# Expand for batch dimension
	combined_mask = combined_mask.unsqueeze(0).expand(batch, num_heads, q_len, kv_len)

	# Combine with attn_mask (logical AND)
	if attn_mask is not None:
		attn_mask = attn_mask & combined_mask
	else:
		attn_mask = combined_mask

	s = s.masked_fill(attn_mask == 0, float('-inf'))
	s = torch.softmax(s, dim=-1)

	out = torch.matmul(s, v)

	return out
