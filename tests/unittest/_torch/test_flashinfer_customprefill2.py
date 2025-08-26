
import torch



def generate_causal_mask(batch_size: int, target_length: int,
						 cache_position: torch.Tensor, device: torch.device):
	causal_mask = torch.arange(
		target_length,
		device=device).unsqueeze(0) <= cache_position.unsqueeze(-1)
	causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

	return causal_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
	"""
	This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
	num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
	"""
	batch, num_key_value_heads, slen, head_dim = hidden_states.shape
	if n_rep == 1:
		return hidden_states
	hidden_states = hidden_states[:, :,
								  None, :, :].expand(batch, num_key_value_heads,
													 n_rep, slen, head_dim)
	return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
								 head_dim)


def scaled_dot_product_attention(q, k, v, attn_mask):
	head_dim = q.size(1)
	s = torch.einsum('bhsd,bhSd->bhsS', q, k)

	s /= head_dim**0.5

	s = s.masked_fill(attn_mask == 0, float('-inf'))
	s = torch.softmax(s, dim=-1)

	out = torch.matmul(s, v)

	return out

def prefill_forward(q, k, v, num_heads, head_dim, num_kv_heads, num_ctx_tokens):
	qq = q[:num_ctx_tokens]
	q_len = qq.shape[0]
	#if q_len > 1000:
	#    return torch.zeros_like(qq).view(q_len, -1)
	qq = qq.view(1, q_len, num_heads, head_dim).transpose(1, 2)

	key_states = k[None].transpose(1, 2).to(q.dtype)
	value_states = v[None].transpose(1, 2).to(q.dtype)

	num_key_value_groups = num_heads // num_kv_heads
	key_states = repeat_kv(key_states, num_key_value_groups)
	value_states = repeat_kv(value_states, num_key_value_groups)

	cache_position = torch.arange(0, q_len, device=q.device)
	attn_mask = generate_causal_mask(1, q_len, cache_position, q.device) if q_len > 1 else None
	#print(f'{key_states.shape=}, {value_states.shape=}')

	#attn_output = torch.nn.functional.scaled_dot_product_attention(
	#	qq,
	#	key_states,
	#	value_states,
	#	is_causal=True,
	#	attn_mask=attn_mask,
	#)
	attn_output = scaled_dot_product_attention(
		qq,
		key_states,
		value_states,
		attn_mask,
	)

	#print(f'{attn_output.shape=}')
	return attn_output.transpose(1, 2).contiguous().view(q_len, -1)


def test_run_forward ():
	data = torch.load('/workspace1/forward_pattern.pt', weights_only=False)

	input = data['input']
	output = data['output']
	#metadata_init = data['metadata_init']

	#print(f'{input.keys()=}')
	#print(f'{input["q"].shape=}')
	#print(f'{output=}')
	q, k, v, num_heads, head_dim, num_kv_heads = input['q'], input['k'], input['v'], input['num_heads'], input['head_dim'], input['num_kv_heads']
	k = k.view(-1, num_kv_heads, head_dim)
	v = v.view(-1, num_kv_heads, head_dim)

	predicted_output = prefill_forward(q, k, v, num_heads, head_dim, num_kv_heads, num_ctx_tokens=12)
	print(f'{predicted_output.shape=}, {predicted_output.dtype=}')
	print(f'{output.shape=}, {output.dtype=}')

	std_diff = (predicted_output - output).pow(2).sum().sqrt() / output.norm()
	print(f'STD diff: {std_diff}')


if __name__ == "__main__":
	test_run_forward()
