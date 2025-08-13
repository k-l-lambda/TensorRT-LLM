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

    @staticmethod
    def no_kv_cache_forward(
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        num_heads: int,
        num_kv_heads: int,
        metadata: AttentionMetadata,
        *,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        This function is used to perform attention without kv cache.
        Args:
            q (torch.Tensor): Query tensor with shape (seq_len, num_heads * head_dim) or (seq_len, (num_heads + 2 * num_kv_heads) * head_dim),
            k (Optional[torch.Tensor]): Key tensor with shape (seq_len, num_heads * head_dim) or None,
            v (Optional[torch.Tensor]): Value tensor with shape (seq_len, num_heads * head_dim) or None,
        """
        #print(f'input: {q.shape=}')
        dummy = torch.zeros_like(q)
        if torch.cuda.is_current_stream_capturing():
            return dummy

        # lazy loading
        #from flash_attn.flash_attn_interface import flash_attn_varlen_func

        head_dim = q.shape[-1]
        is_fused_qkv = False
        if (k is None) or (v is None):
            assert (k is None) or (
                v is None), "Both k and v has to be None if any of them is None"
            is_fused_qkv = True

        if is_fused_qkv:
            q_size = int(head_dim * num_heads / (num_heads + 2 * num_kv_heads))
            kv_size = int(head_dim * num_kv_heads /
                          (num_heads + 2 * num_kv_heads))
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

        #print(f'{max_seqlen_q=}, {max_seqlen_k=}')
        #attn_output_unpad = flash_attn_varlen_func(
        #    q,
        #    k,
        #    v,
        #    cu_seqlens_q,
        #    cu_seqlens_k,
        #    max_seqlen_q,
        #    max_seqlen_k,
        #    dropout_p=0.0,
        #    softmax_scale=None,
        #    causal=attention_mask == PredefinedAttentionMask.CAUSAL,
        #    # window_size=(-1, -1),  # -1 means infinite context window
        #    alibi_slopes=None,
        #    deterministic=False,
        #    return_attn_probs=False,
        #)

        #return attn_output_unpad.reshape(attn_output_unpad.size(0), -1)

        # Q*K by einstein summation
        s = torch.einsum('shd,SHd->hsS', q, k)

        s /= head_dim**0.5
        s = torch.softmax(s, dim=-1)
        #print(f'{s.shape=}')

        # apply causal mask if needed
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            attention_mask = torch.tril(
                torch.ones((max_seqlen_in_batch, max_seqlen_in_batch),
                           device=s.device, dtype=s.dtype),
                diagonal=0)
            attention_mask = attention_mask.unsqueeze(0)
            #print(f'{attention_mask[0, :4, :4]=}')
            s *= attention_mask

        # S*V by einstein summation
        out = torch.einsum('hsS,SHd->shd', s, v)
        #print(f'{out.shape=}')

        return out.reshape(out.size(0), -1)

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: SparseVanillaAttentionMetadata,
                *,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
                **kwargs) -> torch.Tensor:
        # NOTE: WAR for no kv cache attn e.g. BERT,
        # try to separate the kv cache estimation path from no kv cache attn.
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        return SparseVanillaAttention.no_kv_cache_forward(
            q=q,
            k=k,
            v=v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            metadata=metadata,
            attention_mask=attention_mask)
