from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm.models.modeling_utils import QuantConfig

try:
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
except ImportError:
    AttentionMaskConverter = None

from .interface import (AttentionBackend, AttentionMask, AttentionMetadata,
                        PredefinedAttentionMask)



class SparseVanillaAttentionMetadata(AttentionMetadata):

    def prepare(self) -> None:
        # indices of used cache blocks for each sequence
        assert self.request_ids is not None
        self.block_ids_per_seq = self.kv_cache_manager.get_batch_cache_indices(
            self.request_ids) if self.kv_cache_manager is not None else None


class SparseVanillaAttention(AttentionBackend[SparseVanillaAttentionMetadata]):

    Metadata = SparseVanillaAttentionMetadata

    _access_type = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64
    }

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config, **kwargs)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

    def _single_request_update_kv_cache(self, k, v, kv_cache_tensor, seq_len,
                                        cache_idx, cache_position):
        print(f'{kv_cache_tensor.shape=}')
        #print(f'{k.shape=}, {v.shape=}')
        #print(f'{cache_idx=}, {cache_position=}, {seq_len=}')
        k_out = kv_cache_tensor[cache_idx:, 0, :, :, :]
        v_out = kv_cache_tensor[cache_idx:, 1, :, :, :]
        #print(f'{k_out.shape=}, {v_out.shape=}')
        #print(f'{k.dtype.itemsize=}')

        if k is not None and v is not None:
            access_type = self._access_type[k.dtype.itemsize]
            num_kv_heads, head_dim = k_out.shape[-2:]
            k_out = k_out.view(dtype=access_type)
            v_out = v_out.view(dtype=access_type)
            #print(f'1.{k_out.shape=}, {v_out.shape=}')

            pad_len = (32 - (k.size(1) % 32)) % 32
            if pad_len > 0:
                k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            k = k.reshape(-1, 32, num_kv_heads, head_dim)[:k_out.shape[0]]
            v = v.reshape(-1, 32, num_kv_heads, head_dim)[:v_out.shape[0]]
            #print(f'{k.view(dtype=access_type).shape=}, {v.view(dtype=access_type).shape=}')

            indices = torch.arange(k.shape[0], device=k_out.device, dtype=torch.long)
            k_out.index_copy_(0, indices, k.view(dtype=access_type))
            v_out.index_copy_(0, indices, v.view(dtype=access_type))

        return k_out[:, :seq_len, :, :], v_out[:, :seq_len, :, :]

    def single_forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: AttentionMetadata,
        kv_cache_tensor: torch.Tensor,
        cache_idx: int,
        past_seen_token: int,
        *,
        out_scale: torch.Tensor = None,
        mrope_config: Optional[dict] = None,
        attention_window_size: Optional[int] = None,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        #print(f'{metadata.num_contexts=}, {metadata.num_generations=}')
        """
        This function is used to perform attention without kv cache.
        Args:
            q (torch.Tensor): Query tensor with shape (seq_len, num_heads * head_dim) or (seq_len, (num_heads + 2 * num_kv_heads) * head_dim),
            k (Optional[torch.Tensor]): Key tensor with shape (seq_len, num_heads * head_dim) or None,
            v (Optional[torch.Tensor]): Value tensor with shape (seq_len, num_heads * head_dim) or None,
        """
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads

        #print(f'input: {q.shape=}')

        #if out_scale is not None:
        #    print(f'{out_scale.shape=}')
        #print(f'{mrope_config=}')
        #print(f'{attention_window_size=}')

        head_dim = q.shape[-1]
        is_fused_qkv = False
        if (k is None) or (v is None):
            assert (k is None) or (
                v is None), "Both k and v has to be None if any of them is None"
            is_fused_qkv = True

        #print(f'{is_fused_qkv=}, {num_heads=}, {num_kv_heads=}, {head_dim=}')
        if is_fused_qkv:
            q_size = int(head_dim * num_heads / (num_heads + 2 * num_kv_heads))
            kv_size = int(head_dim * num_kv_heads /
                          (num_heads + 2 * num_kv_heads))
            #print(f'{q_size=}, {kv_size=}')
            q, k, v = q.split([q_size, kv_size, kv_size], dim=-1)
        else:
            q_size = head_dim
        head_dim = int(q_size / num_heads)
        q = q.reshape(-1, num_heads, head_dim).contiguous()
        k = k.reshape(-1, num_kv_heads, head_dim).contiguous()
        v = v.reshape(-1, num_kv_heads, head_dim).contiguous()
        assert q.dim() == 3
        assert k.dim() == 3
        assert v.dim() == 3
        seqlens_in_batch = metadata.seq_lens
        assert seqlens_in_batch is not None, "seq_len can not be None for remove padding inputs attention!"
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
            (1, 0)).to(q.device)
        #print(f'{q.shape=}, {k.shape=}, {v.shape=}')
        #print(f'{cu_seqlens=}')

        max_seqlen_q = max_seqlen_k = max_seqlen_in_batch
        cu_seqlens_q = cu_seqlens_k = cu_seqlens

        #return attn_output_unpad.reshape(attn_output_unpad.size(0), -1)

        target_seq_len = past_seen_token
        if k is not None and v is not None:
            kv_len = k.size(0)
            kk = k.view(1, kv_len, num_kv_heads, head_dim)
            vv = v.view(1, kv_len, num_kv_heads, head_dim)
            target_seq_len += kv_len

            if self.quant_config and self.quant_config.layer_quant_mode.has_any_quant():
                qc = self.quant_config
                if qc.layer_quant_mode.has_fp8_kv_cache():
                    assert kv_cache_tensor.dtype == torch.float8_e4m3fn, f"KV cache should have fp8 dtype, but get {kv_cache_tensor.dtype}"
                    kk = kk.to(torch.float8_e4m3fn)
                    vv = vv.to(torch.float8_e4m3fn)
            assert kk.dtype == vv.dtype == kv_cache_tensor.dtype, f"KV cache dtype {kv_cache_tensor.dtype} does not match k/v dtype {kk.dtype}/{vv.dtype}"

        cache_position = torch.arange(past_seen_token, target_seq_len, device=q.device)
        self._single_request_update_kv_cache(kk, vv, kv_cache_tensor, target_seq_len, cache_idx, cache_position)

        # Q*K by einstein summation
        #print(f'{metadata.kv_cache_params=}')
        #print(f'{q.dtype=}, {k.dtype=}, {v.dtype=}')
        s = torch.einsum('shd,SHd->hsS', q, k)
        s /= head_dim**0.5

        # rewrite s computation by torch.matmul
        #s2 = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2))
        #assert torch.allclose(s, s2), f"Mismatch in attention scores: {s.shape} vs {s2.shape}"

        # apply causal mask if needed
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            attention_mask = torch.tril(
                torch.ones((max_seqlen_in_batch, max_seqlen_in_batch),
                           device=s.device, dtype=s.dtype),
                diagonal=0)
            attention_mask = attention_mask.unsqueeze(0)
            #print(f'{attention_mask.shape=}, {attention_mask[0, :4, :4]=}')
            s = s.masked_fill(attention_mask == 0, float('-inf'))

        s = torch.softmax(s, dim=-1)
        #print(f'{s.shape=}')

        # S*V by einstein summation
        #out = torch.einsum('hsS,SHd->shd', s, v)
        out2 = torch.matmul(s, v.transpose(0, 1)).transpose(0, 1)
        #assert torch.allclose(out, out2), f"Mismatch in attention output: {out.shape} vs {out2.shape}"
        #print(f'{out2.shape=}, {out2[0, :2, :10]=}')
        #print(f'{out2.dtype=}')
        out2 = out2.to(torch.float8_e5m2).to(q.dtype)

        return out2.reshape(out2.size(0), -1) #/ out_scale.to(torch.bfloat16)

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: SparseVanillaAttentionMetadata,
                *,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
                **kwargs) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing() or metadata.kv_cache_manager is None:
            dummy = torch.zeros_like(q)
            return dummy

        # NOTE: WAR for no kv cache attn e.g. BERT,
        # try to separate the kv cache estimation path from no kv cache attn.

        past_seen_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        cache_indices = [block_ids[0] for block_ids in metadata.block_ids_per_seq]
        kv_cache_tensor = metadata.kv_cache_manager.get_buffers(self.layer_idx)
        #print(f'{cache_indices=}')

        o = self.single_forward(
            q=q,
            k=k,
            v=v,
            metadata=metadata,
            cache_idx=cache_indices[0],
            kv_cache_tensor=kv_cache_tensor,
            past_seen_token=past_seen_tokens[0],
            attention_mask=attention_mask, **kwargs)
            attention_mask=attention_mask)

        return o
