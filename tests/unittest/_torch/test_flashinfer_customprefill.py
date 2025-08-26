
import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.attention_backend.flashinfer import forward_pattern_impl, FlashInferAttentionMetadata



def test_run_forward ():
	kv_manager_init = torch.load('/workspace1/kv_manager.pt', weights_only=False)
	kv_cache_manager = KVCacheManager(**kv_manager_init)

	data = torch.load('/workspace1/forward_pattern.pt', weights_only=False)

	input = data['input']
	output = data['output']
	metadata_init = data['metadata_init']

	attn_metadata = FlashInferAttentionMetadata(kv_cache_manager=kv_cache_manager, **metadata_init)

	attn_metadata.seq_lens = torch.tensor([input['q'].shape[0]], dtype=torch.int, pin_memory=True)
	attn_metadata.max_seq_len = 131072
	attn_metadata._seq_lens_cuda = attn_metadata.seq_lens.cuda()
	attn_metadata.num_contexts = 1
	attn_metadata.request_ids = [0]
	attn_metadata.prepare()

	real_output = forward_pattern_impl(**input, metadata=attn_metadata)
	print(f'{real_output.shape=}')
	print(f'{output.shape=}')


if __name__ == "__main__":
	test_run_forward()
