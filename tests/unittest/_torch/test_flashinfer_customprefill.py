
import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.attention_backend.flashinfer import forward_pattern_impl, FlashInferAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams



def test_run_forward ():
	kv_manager_init = torch.load('/workspace1/kv_manager.pt', weights_only=False)
	kv_cache_manager = KVCacheManager(**kv_manager_init)

	data = torch.load('/workspace1/forward_pattern.pt', weights_only=False)

	input = data['input']
	output = data['output']
	metadata_init = data['metadata_init']

	seq_len = output.shape[0]

	for k, v in input.items():
		if isinstance(v, torch.Tensor):
			input[k] = v.cuda(0)

	attn_metadata = FlashInferAttentionMetadata(kv_cache_manager=kv_cache_manager,
		kv_cache_params=KVCacheParams(
		use_cache=True,
		num_cached_tokens_per_seq=[0],
	), **metadata_init)

	attn_metadata.seq_lens = torch.tensor([input['q'].shape[0]], dtype=torch.int, pin_memory=True)
	attn_metadata.max_seq_len = 131072
	attn_metadata._seq_lens_cuda = attn_metadata.seq_lens.cuda()
	attn_metadata.num_contexts = 1
	attn_metadata.request_ids = [0]
	kv_cache_manager.add_dummy_requests(attn_metadata.request_ids, [seq_len])
	attn_metadata.prepare()

	import time

	print("Warming up...")
	for i in range(20):
		forward_pattern_impl(**input, metadata=attn_metadata)

	print("Running...")
	time_n = 10000
	start = time.time()
	for i in range(time_n):
		forward_pattern_impl(**input, metadata=attn_metadata)
	end = time.time()
	average_time = (end - start) / time_n
	print(f'Average time per iteration: {average_time} s')
	return
	predicted_output = forward_pattern_impl(**input, metadata=attn_metadata).cpu()
	#print(f'{predicted_output.shape=}')
	#print(f'{output.shape=}')

	std_diff = (predicted_output - output).pow(2).sum().sqrt() / output.norm()
	print(f'STD diff: {std_diff}')


if __name__ == "__main__":
	test_run_forward()
