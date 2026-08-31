from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
import torch.nn.functional as F
import time

from utils.history_selection import begin_history_chunk, share_history_selection

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = torch.empty_like(x)

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # Write directly to pre-allocated output instead of appending to list
        output[i, :seq_len] = x_i.type_as(x)
        output[i, seq_len:] = x[i, seq_len:]

    return output


def causal_rope_apply_positions(x, grid_sizes, freqs, frame_positions):
    """RoPE with explicit (possibly non-contiguous) per-frame temporal positions.

    Same as causal_rope_apply, but instead of a contiguous range
    freqs[0][start_frame:start_frame + f], each frame f uses the temporal
    frequency at freqs[0][frame_positions[f]]. Used by the Hybrid-RoPE ablation
    to rope selected sink/history frames at their TRUE global positions rather
    than the hybrid scheme's compressed contiguous positions.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    frame_positions = frame_positions.to(device=freqs[0].device, dtype=torch.long)
    output = torch.empty_like(x)

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][frame_positions[:f]].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output[i, :seq_len] = x_i.type_as(x)
        output[i, seq_len:] = x[i, seq_len:]

    return output


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 sink_frames=0,
                 hist_frames=0,
                 tail_frames=0,
                 hist_position_idx=-1,
                 num_hist_candidates=5,
                 lambda_redundancy=1.0,
                 contiguous_rope=False,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.sink_frames = sink_frames
        self.hist_frames = hist_frames
        self.tail_frames = tail_frames
        self.hist_position_idx = hist_position_idx
        self.enable_log = False
        self.num_hist_candidates = num_hist_candidates
        self.lambda_redundancy = lambda_redundancy
        # Hybrid-RoPE ablation: when True, rope selected sink/history frames at
        # their TRUE global temporal positions (original contiguous sliding-window
        # RoPE) instead of the hybrid scheme's compressed contiguous positions.
        self.contiguous_rope = contiguous_rope
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.profiling = False
        self._profile_data = {
            "qkv_rope": 0.0,
            "candidate_scoring": 0.0,
            "kv_cache_manage": 0.0,
            "kv_select_rope": 0.0,
            "flash_attn": 0.0,
            "output_proj": 0.0,
        }

    def reset_profile_data(self):
        for k in self._profile_data:
            self._profile_data[k] = 0.0

    @torch.no_grad()
    def _score_candidates_second_half(self, q, kv_cache, middle_start, num_middle_frames, frame_seqlen):
        if num_middle_frames <= 1:
            return None

        second_half_start = num_middle_frames // 2
        second_half_length = num_middle_frames - second_half_start
        if second_half_length <= 0:
            return None

        actual_num_candidates = min(self.num_hist_candidates, second_half_length)
        if actual_num_candidates == 1:
            return torch.tensor([second_half_start], device=q.device, dtype=torch.long)

        candidate_frame_indices = torch.linspace(
            second_half_start, num_middle_frames - 1, actual_num_candidates + 1,
            device=q.device
        ).long().unique()[:-1]

        mean_q = q.mean(dim=1)  # [B, H, D]

        candidate_mean_k = []
        for frame_idx in candidate_frame_indices:
            start_token = middle_start + frame_idx * frame_seqlen
            end_token = start_token + frame_seqlen
            frame_k = kv_cache["k"][:, start_token:end_token]  # [B, frame_seqlen, H, D]
            candidate_mean_k.append(frame_k.mean(dim=1))  # [B, H, D]

        candidate_mean_k = torch.stack(candidate_mean_k, dim=1)  # [B, C, H, D]

        scores = torch.einsum('bhd,bchd->bch', mean_q.float(), candidate_mean_k.float())
        scores = scores / (self.head_dim ** 0.5)
        scores = scores.mean(dim=-1)  # [B, C]

        num_to_select = min(self.hist_frames, len(candidate_frame_indices))
        _, topk_indices = scores[0].topk(num_to_select)  # [num_to_select]
        selected = candidate_frame_indices[topk_indices].sort().values
        return selected

    @torch.no_grad()
    def _score_candidates_second_half_relaxed(self, kv_cache, middle_start, num_middle_frames,
                                              frame_seqlen, sink_tokens, tail_tokens,
                                              local_start_index, lambda_redundancy):
        if num_middle_frames <= 1:
            return None

        second_half_start = num_middle_frames // 2
        second_half_length = num_middle_frames - second_half_start
        if second_half_length <= 0:
            return None

        actual_num_candidates = min(self.num_hist_candidates, second_half_length)
        if actual_num_candidates <= 0:
            return None
        if actual_num_candidates == 1:
            return torch.tensor([second_half_start], device=kv_cache["k"].device, dtype=torch.long)

        candidate_frame_indices = torch.linspace(
            second_half_start, num_middle_frames - 1, actual_num_candidates + 1,
            device=kv_cache["k"].device
        ).long().unique()[:-1]

        k_cache = kv_cache["k"]

        def _proto(k_tokens):
            return F.normalize(k_tokens.mean(dim=1).mean(dim=1).float(), dim=-1)

        sink_proto = _proto(k_cache[:, :sink_tokens]) if sink_tokens > 0 else None
        tail_proto = (_proto(k_cache[:, local_start_index - tail_tokens:local_start_index])
                    if tail_tokens > 0 else None)

        cand_protos = []
        for frame_idx in candidate_frame_indices:
            start_token = middle_start + frame_idx * frame_seqlen
            end_token = start_token + frame_seqlen
            cand_protos.append(_proto(k_cache[:, start_token:end_token]))
        cand_protos = torch.stack(cand_protos, dim=1)  # [B, C, D]

        scores = torch.zeros(cand_protos.shape[0], cand_protos.shape[1],
                            device=cand_protos.device)
        if sink_proto is not None:
            scores = scores + (cand_protos * sink_proto[:, None, :]).sum(dim=-1)
        if tail_proto is not None:
            scores = scores - lambda_redundancy * (cand_protos * tail_proto[:, None, :]).sum(dim=-1)

        num_to_select = min(self.hist_frames, candidate_frame_indices.numel())
        topk = scores.topk(num_to_select, dim=1).indices
        selected = candidate_frame_indices[topk[0]].sort().values
        return selected

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        block_index=0
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        _profiling = self.profiling
        if _profiling:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

        if _profiling:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            self._profile_data["qkv_rope"] += _t1 - _t0

        sink_tokens = self.sink_frames * frame_seqlen
        hist_tokens = self.hist_frames * frame_seqlen
        tail_tokens = self.tail_frames * frame_seqlen

        current_end = current_start + roped_query.shape[1]
        num_new_tokens = roped_query.shape[1]
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens

        # ========================
        # === Caching KV Mechanism
        # ========================
        if _profiling:
            torch.cuda.synchronize()
            _t_cache0 = time.perf_counter()
        # we first check if we need to evict tokens to overflow the cache
        cache_size = kv_cache["k"].shape[1]
        if local_end_index > cache_size:
            # we keep tokens based on local attention window size, minus room for new tokens
            keep_tokens = min(self.local_attn_size * frame_seqlen, cache_size) - num_new_tokens
            keep_tokens = max(0, keep_tokens)  # Ensure non-negative
            old_local_end = kv_cache["local_end_index"].item()
            if old_local_end > 0 and keep_tokens > 0:
                # Calculate how many tokens we actually have to keep
                actual_keep = min(keep_tokens, old_local_end)
                # Preserve sink frames - never evict the first sink_tokens
                if sink_tokens > 0 and old_local_end > sink_tokens:
                    # Keep sink frames at positions 0:sink_tokens (don't touch them)
                    # Shift the remaining tokens from the tail to positions after sink
                    non_sink_keep = actual_keep - sink_tokens
                    if non_sink_keep > 0:
                        kv_cache["k"][:, sink_tokens:sink_tokens + non_sink_keep] = \
                            kv_cache["k"][:, old_local_end - non_sink_keep:old_local_end].clone()
                        kv_cache["v"][:, sink_tokens:sink_tokens + non_sink_keep] = \
                            kv_cache["v"][:, old_local_end - non_sink_keep:old_local_end].clone()
                    # Update indices - new tokens go right after kept tokens
                    local_start_index = actual_keep
                    local_end_index = actual_keep + num_new_tokens
                else:
                    # No sink frames or not enough data yet - use original logic
                    kv_cache["k"][:, :actual_keep] = kv_cache["k"][:, old_local_end - actual_keep:old_local_end].clone()
                    kv_cache["v"][:, :actual_keep] = kv_cache["v"][:, old_local_end - actual_keep:old_local_end].clone()

                    # Update indices - new tokens go right after kept tokens
                    local_start_index = actual_keep
                    local_end_index = actual_keep + num_new_tokens
            else:
                # No tokens to keep or cache empty, start fresh
                local_start_index = 0
                local_end_index = num_new_tokens
        # we then cache un-roped key and value
        kv_cache["k"][:, local_start_index:local_end_index] = k
        kv_cache["v"][:, local_start_index:local_end_index] = v

        if _profiling:
            torch.cuda.synchronize()
            _t_cache1 = time.perf_counter()
            self._profile_data["kv_cache_manage"] += _t_cache1 - _t_cache0

        # ====================================
        # === Select KV from Cache, Apply RoPE
        # ====================================
        # if the cache is not enough for a attention window, we use all the cache
        # if local_start_index <= sink_tokens + hist_tokens + tail_tokens:
        if local_start_index <= sink_tokens + frame_seqlen + tail_tokens:
            if _profiling:
                torch.cuda.synchronize()
                _t_sel0 = time.perf_counter()
            cache_grid_sizes = grid_sizes.clone()
            cache_grid_sizes[:, 0] = local_end_index // frame_seqlen
            global_start_frame = (current_end - local_end_index) // frame_seqlen
            rope_key = causal_rope_apply(
                kv_cache["k"][:, :local_end_index], cache_grid_sizes, freqs, start_frame=global_start_frame
            ).type_as(v)
            input_v = kv_cache["v"][:, :local_end_index]
            input_key = rope_key
            if _profiling:
                torch.cuda.synchronize()
                _t_sel1 = time.perf_counter()
                self._profile_data["kv_select_rope"] += _t_sel1 - _t_sel0
            # Apply Attention
            if _profiling:
                torch.cuda.synchronize()
                _t_attn0 = time.perf_counter()
            x = attention(roped_query, input_key, input_v)
            if _profiling:
                torch.cuda.synchronize()
                _t_attn1 = time.perf_counter()
                self._profile_data["flash_attn"] += _t_attn1 - _t_attn0
        # if the cache is enough for a attention window, we use the selected key and value from the cache
        else:
            # After eviction, local_end_index != global position
            # Use current_end for correct RoPE positions

            # Select Sink from Cache
            # First n frames (indices 0 to sink_tokens) in the cache
            sink_k = kv_cache["k"][:, :sink_tokens]
            sink_v = kv_cache["v"][:, :sink_tokens]
            sink_indices = torch.arange(0, sink_tokens, device=kv_cache["k"].device)
            sink_frames_before_adjustment = (sink_indices // frame_seqlen).unique()

            # Select History from Cache
            # Sample n frames from the middle region (between sink and tail) in the cache
            # Middle region: from sink_tokens to (local_start_index - tail_tokens)
            middle_start = sink_tokens
            middle_end = local_start_index - tail_tokens
            middle_length = middle_end - middle_start
            # Sample hist_tokens uniformly from middle region
            # We sample frame by frame to maintain frame boundaries
            num_hist_frames = hist_tokens // frame_seqlen
            num_middle_frames = middle_length // frame_seqlen

            # Check for precomputed attention-based selection (from layer 0)
            selected_hist = kv_cache.get("selected_hist_frame_idx", None)

            if selected_hist is not None:
                sampled_frame_indices = selected_hist
                num_hist_frames = len(selected_hist)
            elif block_index == 0 and self.hist_position_idx == -2:
                if _profiling:
                    torch.cuda.synchronize()
                    _t_score0 = time.perf_counter()
                selected_hist = self._score_candidates_second_half_relaxed(
                    kv_cache, middle_start, num_middle_frames, frame_seqlen,
                    sink_tokens, tail_tokens, local_start_index, self.lambda_redundancy
                )
                if _profiling:
                    torch.cuda.synchronize()
                    _t_score1 = time.perf_counter()
                    self._profile_data["candidate_scoring"] += _t_score1 - _t_score0
                if selected_hist is not None:
                    kv_cache["selected_hist_frame_idx"] = selected_hist
                    sampled_frame_indices = selected_hist
                    num_hist_frames = len(selected_hist)
                else:
                    sampled_frame_indices = torch.arange(
                        num_middle_frames, device=kv_cache["k"].device
                    )
                    num_hist_frames = num_middle_frames
            elif num_middle_frames < num_hist_frames:
                if self.hist_position_idx >= 0:
                    num_candidates = min(num_middle_frames, num_hist_frames)
                    candidate_frame_indices = torch.linspace(
                        0, num_middle_frames - 1, num_candidates + 2,
                        device=kv_cache["k"].device
                    ).long()[1:-1]
                    if len(candidate_frame_indices) == 0:
                        candidate_frame_indices = torch.tensor(
                            [(num_middle_frames - 1) // 2], device=kv_cache["k"].device, dtype=torch.long
                        )
                    idx = min(self.hist_position_idx, len(candidate_frame_indices) - 1)
                    sampled_frame_indices = candidate_frame_indices[idx:idx+1]
                    num_hist_frames = 1
                else:
                    sampled_frame_indices = torch.arange(
                        num_middle_frames, device=kv_cache["k"].device
                    )
                    num_hist_frames = num_middle_frames
            else:
                num_candidates = num_hist_frames
                candidate_frame_indices = torch.linspace(
                    0, num_middle_frames - 1, num_candidates + 2,
                    device=kv_cache["k"].device
                ).long()[1:-1]
                if self.hist_position_idx >= 0:
                    sampled_frame_indices = candidate_frame_indices[self.hist_position_idx:self.hist_position_idx + 1]
                    num_hist_frames = 1
                else:
                    sampled_frame_indices = candidate_frame_indices

            if _profiling:
                torch.cuda.synchronize()
                _t_sel0 = time.perf_counter()

            hist_indices = []
            if num_hist_frames > 0:
                for frame_idx in sampled_frame_indices:
                    start_token = middle_start + frame_idx * frame_seqlen
                    end_token = start_token + frame_seqlen
                    hist_indices.append(torch.arange(start_token, end_token, device=kv_cache["k"].device))
                hist_indices = torch.cat(hist_indices)
                hist_k = kv_cache["k"][:, hist_indices]
                hist_v = kv_cache["v"][:, hist_indices]
            else:
                hist_indices = torch.tensor([], dtype=torch.long, device=kv_cache["k"].device)
                hist_k = kv_cache["k"][:, :0]
                hist_v = kv_cache["v"][:, :0]
            local_token_positions = middle_start + sampled_frame_indices * frame_seqlen
            global_token_positions = current_end - (local_end_index - local_token_positions)
            hist_frames_before_adjustment = (global_token_positions // frame_seqlen).unique()

            # Select Tail from Cache
            # Last n frames BEFORE the current block in the cache
            tail_k = kv_cache["k"][:, local_start_index - tail_tokens:local_start_index]
            tail_v = kv_cache["v"][:, local_start_index - tail_tokens:local_start_index]
            tail_indices = torch.arange(local_start_index - tail_tokens, local_start_index, device=kv_cache["k"].device)
            tail_frames_before_adjustment = (tail_indices // frame_seqlen).unique()

            # Select Current from Cache
            # The block being denoised right now in the cache
            current_k = kv_cache["k"][:, local_start_index:local_end_index]
            current_v = kv_cache["v"][:, local_start_index:local_end_index]
            current_num_frames = num_new_tokens // frame_seqlen
            current_frames_before_adjustment = torch.arange(
                                                    current_start_frame,
                                                    current_start_frame + current_num_frames,
                                                    device=kv_cache["k"].device
                                                )
            # Apply RoPE
            # For tail: use real/absolute position
            tail_num_frames = tail_tokens // frame_seqlen
            if tail_tokens > 0:
                tail_start_frame = current_start_frame - tail_num_frames  # Tail ends right before current block
                tail_grid_sizes = grid_sizes.clone()
                tail_grid_sizes[:, 0] = tail_num_frames
                roped_tail_k = causal_rope_apply(
                    tail_k, tail_grid_sizes, freqs, start_frame=tail_start_frame
                ).type_as(v)
            else:
                roped_tail_k = kv_cache["k"][:, :0]
                tail_start_frame = current_start_frame  # No tail, sink/hist positioned before current
            # For sink and history: use relative position (rope them as if they are right before current query)
            # This means we use position indices relative to current frame
            sink_hist_k = torch.cat([sink_k, hist_k], dim=1)
            sink_hist_v = torch.cat([sink_v, hist_v], dim=1)
            sink_hist_num_frames = sink_hist_k.shape[1] // frame_seqlen
            if sink_hist_num_frames > 0:
                sink_hist_grid_sizes = grid_sizes.clone()
                sink_hist_grid_sizes[:, 0] = sink_hist_num_frames
                if self.contiguous_rope:
                    # Hybrid-RoPE ablation baseline: rope sink + history at their
                    # TRUE global temporal positions (original contiguous
                    # sliding-window RoPE), not the compressed relative positions.
                    sink_num_frames = sink_tokens // frame_seqlen
                    sink_global_frames = torch.arange(
                        0, sink_num_frames, device=kv_cache["k"].device
                    )
                    hist_global_frames = (global_token_positions // frame_seqlen)
                    sink_hist_positions = torch.cat(
                        [sink_global_frames, hist_global_frames]
                    )
                    roped_sink_hist_k = causal_rope_apply_positions(
                        sink_hist_k, sink_hist_grid_sizes, freqs, sink_hist_positions
                    ).type_as(v)
                    sink_hist_frames_indices_after_rope_adj = sink_hist_positions
                else:
                    sink_hist_start_frame = tail_start_frame - sink_hist_num_frames
                    roped_sink_hist_k = causal_rope_apply(
                        sink_hist_k, sink_hist_grid_sizes, freqs, start_frame=sink_hist_start_frame
                    ).type_as(v)
                    sink_hist_frames_indices_after_rope_adj = torch.arange(
                        sink_hist_start_frame,
                        sink_hist_start_frame + sink_hist_num_frames,
                        device=kv_cache["k"].device
                    )
            else:
                roped_sink_hist_k = sink_hist_k  # Empty tensor, no RoPE needed
                sink_hist_frames_indices_after_rope_adj = torch.tensor(
                    [],
                    dtype=torch.long,
                    device=kv_cache["k"].device
                )
            # For current: use real/absolute position
            current_grid_sizes = grid_sizes.clone()
            current_grid_sizes[:, 0] = current_num_frames
            roped_current_k = causal_rope_apply(
                current_k, current_grid_sizes, freqs, start_frame=current_start_frame
            ).type_as(v)

            # Concatenate all parts
            # Order: sink + history + tail + current
            parts_k = [roped_sink_hist_k]
            parts_v = [sink_hist_v]
            parts_k.append(roped_tail_k)
            parts_v.append(tail_v)
            parts_k.append(roped_current_k)
            parts_v.append(current_v)
            input_key = torch.cat(parts_k, dim=1)
            input_v = torch.cat(parts_v, dim=1)

            if _profiling:
                torch.cuda.synchronize()
                _t_sel1 = time.perf_counter()
                self._profile_data["kv_select_rope"] += _t_sel1 - _t_sel0

            # Apply Attention
            if _profiling:
                torch.cuda.synchronize()
                _t_attn0 = time.perf_counter()
            x = attention(roped_query, input_key, input_v)
            if _profiling:
                torch.cuda.synchronize()
                _t_attn1 = time.perf_counter()
                self._profile_data["flash_attn"] += _t_attn1 - _t_attn0

            # Log sink_hist_frames_indices_after_rope_adj and hist_frames_before_adjustment
            # into text file named "log.txt"
            # if self.enable_log:
            #     final_frames = (
            #         sink_hist_frames_indices_after_rope_adj.tolist() +
            #         tail_frames_before_adjustment.tolist() +
            #         current_frames_before_adjustment.tolist()
            #     )
            #     with open("log.txt", "a") as f:
            #         f.write(f"sink_frames_before_adjustment   : {sink_frames_before_adjustment.tolist()}\n")
            #         f.write(f"hist_frames_before_adjustment   : {hist_frames_before_adjustment.tolist()}\n")
            #         f.write(f"tail_frames_before_adjustment   : {tail_frames_before_adjustment.tolist()}\n")
            #         f.write(f"current_frames_before_adjustment: {current_frames_before_adjustment.tolist()}\n")
            #         f.write(f"***\n")
            #         f.write(f"final_frames_after_adjustment   : {final_frames}\n")
            #         f.write(f"--------------------------------\n")

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        if _profiling:
            torch.cuda.synchronize()
            _t_out0 = time.perf_counter()
        x = x.flatten(2)
        x = self.o(x)
        if _profiling:
            torch.cuda.synchronize()
            _t_out1 = time.perf_counter()
            self._profile_data["output_proj"] += _t_out1 - _t_out0
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 sink_frames=0,
                 hist_frames=0,
                 tail_frames=0,
                 hist_position_idx=-1,
                 num_hist_candidates=5,
                 lambda_redundancy=1.0,
                 contiguous_rope=False,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size,
            sink_size, sink_frames, hist_frames, tail_frames,
            hist_position_idx,
            num_hist_candidates,
            lambda_redundancy=lambda_redundancy,
            contiguous_rope=contiguous_rope,
            qk_norm=qk_norm, eps=eps
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        timestep=None,
        block_index=0
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            block_index=block_index
        )

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 sink_frames=0,
                 hist_frames=0,
                 tail_frames=0,
                 hist_position_idx=-1,
                 num_hist_candidates=5,
                 lambda_redundancy=1.0,
                 contiguous_rope=False,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        self.sink_frames = sink_frames
        self.hist_frames = hist_frames
        self.tail_frames = tail_frames

        # Print KV cache config once (not per block)
        print("*"*60)
        print(f"KV cache config: kv_cache_size={local_attn_size}, sink_frames={sink_frames}, hist_frames={hist_frames}, tail_frames={tail_frames}")

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                cross_attn_type,
                dim,
                ffn_dim,
                num_heads,
                local_attn_size,
                sink_size,
                sink_frames,
                hist_frames,
                tail_frames,
                hist_position_idx,
                num_hist_candidates,
                lambda_redundancy,
                contiguous_rope,
                qk_norm,
                cross_attn_norm,
                eps
            )
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            timestep=t
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # Select History once per autoregressive chunk. All denoising passes for
        # the same current_start reuse it; advancing current_start invalidates it.
        begin_history_chunk(kv_cache, current_start)

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "block_index": block_index
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
                if block_index == 0:
                    share_history_selection(kv_cache)
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "block_index": block_index
                    }
                )
                x = block(x, **kwargs)
                if block_index == 0:
                    share_history_selection(kv_cache)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


                # # ========= Uniform sampling starts (first half) =========
                # sampled_frame_indices = torch.linspace(
                #     0, (num_middle_frames - 1) // 2, num_hist_frames,
                #     device=kv_cache["k"].device
                # ).long()
                # # ========= Uniform sampling ends (first half) =========

                # # ========= Uniform sampling starts (second half) =========
                # sampled_frame_indices = torch.linspace(
                #     (num_middle_frames - 1) // 2, num_middle_frames - 1, num_hist_frames,
                #     device=kv_cache["k"].device
                # ).long()
                # # ========= Uniform sampling ends =========

                # +++ Exponential sampling starts +++
                # Exponential sampling: earlier frames are sampled more densely
                # exp_weights = torch.exp(torch.linspace(0, 1, num_hist_frames, device=kv_cache["k"].device)) - 1
                # exp_weights = exp_weights / exp_weights[-1]
                # sampled_frame_indices = ((1 - exp_weights) * (num_middle_frames - 1)).long()
                # sampled_frame_indices = sampled_frame_indices.flip(0)  # Ensure ascending order
                # +++ Exponential sampling ends +++

                # Importance-based sampling: compute attention between mean Q and history K
                # +++ Importance-based sampling starts +++
                # q shape: [B, num_new_tokens, H, D] - current query
                # Compute mean query across tokens: [B, H, D]
                # mean_q = q.mean(dim=1)

                # # Get middle region keys from cache: [B, middle_length, H, D]
                # middle_k = kv_cache["k"][:, middle_start:middle_end]

                # # Reshape to frame level: [B, num_middle_frames, frame_seqlen, H, D]
                # middle_k_frames = middle_k.view(
                #     middle_k.shape[0], num_middle_frames, frame_seqlen,
                #     middle_k.shape[2], middle_k.shape[3]
                # )

                # # Mean key per frame: [B, num_middle_frames, H, D]
                # mean_k_per_frame = middle_k_frames.mean(dim=2)

                # # Compute attention scores: Q @ K^T / sqrt(d)
                # # mean_q: [B, H, D] -> [B, H, 1, D]
                # # mean_k_per_frame: [B, num_middle_frames, H, D] -> [B, H, num_middle_frames, D]
                # mean_q_expanded = mean_q.unsqueeze(2)  # [B, H, 1, D]
                # mean_k_transposed = mean_k_per_frame.permute(0, 2, 1, 3)  # [B, H, num_middle_frames, D]

                # # Attention scores: [B, H, 1, num_middle_frames]
                # scale = mean_q.shape[-1] ** -0.5
                # attn_scores = torch.matmul(mean_q_expanded, mean_k_transposed.transpose(-2, -1)) * scale

                # # Average across batch and heads: [num_middle_frames]
                # attn_scores = attn_scores.squeeze(2).mean(dim=(0, 1))  # [num_middle_frames]

                # # Select top-k frames with highest attention scores
                # _, sampled_frame_indices = torch.topk(attn_scores, num_hist_frames)
                # sampled_frame_indices = sampled_frame_indices.sort().values
                # +++ Importance-based sampling ends +++


                # # ================================
                # # === Partial Multi-Head Grouped Attention ===
                # # ================================
                # # Most heads see full context (preserve original capability)
                # # Some heads are specialized for specific temporal regions

                # full_head_indices = self.full_head_indices    # e.g., [0..5] - 6 heads
                # tail_head_indices = self.tail_head_indices    # e.g., [6, 7] - 2 heads
                # hist_head_indices = self.hist_head_indices    # e.g., [8, 9] - 2 heads
                # sink_head_indices = self.sink_head_indices    # e.g., [10, 11] - 2 heads

                # # Build full context for full-attention heads
                # if tail_tokens > 0:
                #     full_key = torch.cat([roped_sink_k, roped_hist_k, roped_tail_k], dim=1)
                #     full_v = torch.cat([sink_v, hist_v, tail_v], dim=1)
                # else:
                #     full_key = torch.cat([roped_sink_k, roped_hist_k], dim=1)
                #     full_v = torch.cat([sink_v, hist_v], dim=1)

                # # === Full context heads (main workhorses) ===
                # q_full = roped_query[:, :, full_head_indices, :]
                # k_full = full_key[:, :, full_head_indices, :]
                # v_full = full_v[:, :, full_head_indices, :]
                # x_full = attention(q_full, k_full, v_full)

                # # === Specialized tail heads ===
                # q_tail = roped_query[:, :, tail_head_indices, :]
                # if tail_tokens > 0:
                #     k_tail = roped_tail_k[:, :, tail_head_indices, :]
                #     v_tail = tail_v[:, :, tail_head_indices, :]
                # else:
                #     # Fallback to full context if no tail
                #     k_tail = full_key[:, :, tail_head_indices, :]
                #     v_tail = full_v[:, :, tail_head_indices, :]
                # x_tail = attention(q_tail, k_tail, v_tail)

                # # === Specialized history heads ===
                # q_hist = roped_query[:, :, hist_head_indices, :]
                # if hist_tokens > 0:
                #     k_hist = roped_hist_k[:, :, hist_head_indices, :]
                #     v_hist = hist_v[:, :, hist_head_indices, :]
                # else:
                #     # Fallback to full context if no history
                #     k_hist = full_key[:, :, hist_head_indices, :]
                #     v_hist = full_v[:, :, hist_head_indices, :]
                # x_hist = attention(q_hist, k_hist, v_hist)

                # # === Specialized sink heads ===
                # q_sink = roped_query[:, :, sink_head_indices, :]
                # if sink_tokens > 0:
                #     k_sink = roped_sink_k[:, :, sink_head_indices, :]
                #     v_sink = sink_v[:, :, sink_head_indices, :]
                # else:
                #     # Fallback to full context if no sink
                #     k_sink = full_key[:, :, sink_head_indices, :]
                #     v_sink = full_v[:, :, sink_head_indices, :]
                # x_sink = attention(q_sink, k_sink, v_sink)

                # # === Reassemble all heads ===
                # x = torch.empty(
                #     roped_query.shape[0], roped_query.shape[1],
                #     self.num_heads, self.head_dim,
                #     device=roped_query.device, dtype=roped_query.dtype
                # )
                # x[:, :, full_head_indices, :] = x_full
                # x[:, :, tail_head_indices, :] = x_tail
                # x[:, :, hist_head_indices, :] = x_hist
                # x[:, :, sink_head_indices, :] = x_sink

                # # ================================
                # # === Partial Multi-Head Grouped Attention ===
                # # ================================
                # # Most heads see full context (preserve original capability)
                # # Some heads are specialized for specific temporal regions

                # full_head_indices = self.full_head_indices    # e.g., [0..5] - 6 heads
                # tail_head_indices = self.tail_head_indices    # e.g., [6, 7] - 2 heads
                # hist_head_indices = self.hist_head_indices    # e.g., [8, 9] - 2 heads
                # sink_head_indices = self.sink_head_indices    # e.g., [10, 11] - 2 heads

                # # Build full context for full-attention heads
                # # Use the new combined roped_sink_hist_k
                # if tail_tokens > 0:
                #     full_key = torch.cat([roped_sink_hist_k, roped_tail_k], dim=1)
                #     full_v = torch.cat([sink_hist_v, tail_v], dim=1)
                # else:
                #     full_key = roped_sink_hist_k
                #     full_v = sink_hist_v

                # # === Full context heads (main workhorses) ===
                # q_full = roped_query[:, :, full_head_indices, :]
                # k_full = full_key[:, :, full_head_indices, :]
                # v_full = full_v[:, :, full_head_indices, :]
                # x_full = attention(q_full, k_full, v_full)

                # # === Specialized tail heads ===
                # q_tail = roped_query[:, :, tail_head_indices, :]
                # if tail_tokens > 0:
                #     k_tail = roped_tail_k[:, :, tail_head_indices, :]
                #     v_tail = tail_v[:, :, tail_head_indices, :]
                # else:
                #     # Fallback to full context if no tail
                #     k_tail = full_key[:, :, tail_head_indices, :]
                #     v_tail = full_v[:, :, tail_head_indices, :]
                # x_tail = attention(q_tail, k_tail, v_tail)

                # # === Specialized history heads ===
                # # History heads see history + tail (adds recent context to reduce drift)
                # q_hist = roped_query[:, :, hist_head_indices, :]
                # if hist_tokens > 0:
                #     # History is the second part of sink_hist (after sink_tokens)
                #     k_hist_only = roped_sink_hist_k[:, sink_tokens:, hist_head_indices, :]
                #     v_hist_only = sink_hist_v[:, sink_tokens:, hist_head_indices, :]
                #     if tail_tokens > 0:
                #         # Concat history + tail for better temporal continuity
                #         k_hist = torch.cat([k_hist_only, roped_tail_k[:, :, hist_head_indices, :]], dim=1)
                #         v_hist = torch.cat([v_hist_only, tail_v[:, :, hist_head_indices, :]], dim=1)
                #     else:
                #         k_hist = k_hist_only
                #         v_hist = v_hist_only
                # else:
                #     # Fallback to full context if no history
                #     k_hist = full_key[:, :, hist_head_indices, :]
                #     v_hist = full_v[:, :, hist_head_indices, :]
                # x_hist = attention(q_hist, k_hist, v_hist)

                # # === Specialized sink heads ===
                # # Sink heads see sink + tail (maintains appearance-to-current connection)
                # q_sink = roped_query[:, :, sink_head_indices, :]
                # if sink_tokens > 0:
                #     # Sink is the first part of sink_hist (0 to sink_tokens)
                #     k_sink_only = roped_sink_hist_k[:, :sink_tokens, sink_head_indices, :]
                #     v_sink_only = sink_hist_v[:, :sink_tokens, sink_head_indices, :]
                #     if tail_tokens > 0:
                #         # Concat sink + tail for appearance consistency
                #         k_sink = torch.cat([k_sink_only, roped_tail_k[:, :, sink_head_indices, :]], dim=1)
                #         v_sink = torch.cat([v_sink_only, tail_v[:, :, sink_head_indices, :]], dim=1)
                #     else:
                #         k_sink = k_sink_only
                #         v_sink = v_sink_only
                # else:
                #     # Fallback to full context if no sink
                #     k_sink = full_key[:, :, sink_head_indices, :]
                #     v_sink = full_v[:, :, sink_head_indices, :]
                # x_sink = attention(q_sink, k_sink, v_sink)

                # # === Reassemble all heads ===
                # x = torch.empty(
                #     roped_query.shape[0], roped_query.shape[1],
                #     self.num_heads, self.head_dim,
                #     device=roped_query.device, dtype=roped_query.dtype
                # )
                # x[:, :, full_head_indices, :] = x_full
                # x[:, :, tail_head_indices, :] = x_tail
                # x[:, :, hist_head_indices, :] = x_hist
                # x[:, :, sink_head_indices, :] = x_sink

                # # ================================
                # # === Log final frame order ===
                # # ================================
                # # Final frame order: sink + history + tail + current
                # current_frame_indices = torch.arange(0, current_num_frames, device=kv_cache["k"].device)
                # final_frame_indices = torch.cat([sink_frames_before_adjustment,
                #                                  hist_frames_before_adjustment,
                #                                  tail_frames_before_adjustment,
                #                                  current_frame_indices])
                # # Log with GLOBAL frame indices for verification
                # # Calculate global offset: how many frames were evicted
                # global_tail_end_frame = current_start_frame  # Tail ends at current block start
                # local_tail_end_frame = local_start_index // frame_seqlen
                # frame_offset = global_tail_end_frame - local_tail_end_frame  # Offset due to eviction
                # global_frame_indices = final_frame_indices + frame_offset
                # with open("final_key_indices.txt", "a") as f:
                #     f.write(f"Frame Indices Before Eviction: {','.join(map(str, final_frame_indices.cpu().tolist()))}\n")
                #     f.write(f"Frame Indices After Eviction: {','.join(map(str, global_frame_indices.cpu().tolist()))}\n")
                #     f.write(f"Frame Indices After RoPE Adjustment: {','.join(map(str, sink_hist_frames_indices_after_rope_adj.cpu().tolist()))}\n")
                #     f.write("--------------------------------\n")

                # # ================================
                # # === Apply RoPE ===
                # # ================================
                # # Calculate frame offset for converting local to global positions
                # local_tail_end_frame = local_end_index // frame_seqlen
                # frame_offset = global_end_frame - local_tail_end_frame  # Offset due to eviction
                # # For tail: use real/absolute position
                # tail_num_frames = tail_tokens // frame_seqlen
                # if tail_tokens > 0:
                #     tail_start_frame = global_end_frame - tail_num_frames  # Use global position
                #     tail_grid_sizes = grid_sizes.clone()
                #     tail_grid_sizes[:, 0] = tail_num_frames
                #     roped_tail_k = causal_rope_apply(
                #         tail_k, tail_grid_sizes, freqs, start_frame=tail_start_frame
                #     ).type_as(v)
                #     tail_rope_frames = torch.arange(
                #         tail_start_frame,
                #         tail_start_frame + tail_num_frames,
                #         device=kv_cache["k"].device
                #     )
                # else:
                #     roped_tail_k = None
                #     tail_start_frame = global_end_frame  # Use global position
                #     tail_rope_frames = torch.tensor([], dtype=torch.long, device=kv_cache["k"].device)
                # # For sink and history: use real/absolute position
                # # Calculate absolute frame positions for sink frames
                # sink_num_frames = sink_tokens // frame_seqlen
                # if sink_num_frames > 0:
                #     # Sink frames are at the beginning of cache, so their absolute positions are frame_offset + local indices
                #     sink_absolute_frames = frame_offset + sink_frames_before_adjustment
                #     sink_start_frame = sink_absolute_frames.min().item()
                #     sink_grid_sizes = grid_sizes.clone()
                #     sink_grid_sizes[:, 0] = sink_num_frames
                #     roped_sink_k = causal_rope_apply(
                #         sink_k, sink_grid_sizes, freqs, start_frame=sink_start_frame
                #     ).type_as(v)
                #     sink_rope_frames = sink_absolute_frames
                # else:
                #     roped_sink_k = sink_k
                #     sink_rope_frames = torch.tensor([], dtype=torch.long, device=kv_cache["k"].device)

                # # Calculate absolute frame positions for history frames
                # num_hist_frames = hist_tokens // frame_seqlen
                # if num_hist_frames > 0:
                #     # History frames are sampled from middle region
                #     # Convert local middle region frame indices to absolute positions
                #     middle_start_frame = (middle_start // frame_seqlen) + frame_offset
                #     hist_absolute_frames = middle_start_frame + hist_frames_before_adjustment
                #     hist_start_frame = hist_absolute_frames.min().item()
                #     hist_grid_sizes = grid_sizes.clone()
                #     hist_grid_sizes[:, 0] = num_hist_frames
                #     roped_hist_k = causal_rope_apply(
                #         hist_k, hist_grid_sizes, freqs, start_frame=hist_start_frame
                #     ).type_as(v)
                #     hist_rope_frames = hist_absolute_frames
                # else:
                #     roped_hist_k = hist_k
                #     hist_rope_frames = torch.tensor([], dtype=torch.long, device=kv_cache["k"].device)
                # # Concatenate roped sink and history
                # sink_hist_k = torch.cat([roped_sink_k, roped_hist_k], dim=1)
                # sink_hist_v = torch.cat([sink_v, hist_v], dim=1)
                # sink_hist_frames_indices_after_rope_adj = torch.cat([
                #     sink_rope_frames,
                #     hist_rope_frames
                # ])
                # # ================================
                # # === Concatenate all parts ===
                # # ================================
                # # Order: sink + history + tail
                # if tail_tokens > 0:
                #     input_key = torch.cat([sink_hist_k, roped_tail_k], dim=1)
                #     input_v = torch.cat([sink_hist_v, tail_v], dim=1)
                # else:
                #     input_key = sink_hist_k
                #     input_v = sink_hist_v
                # # ================================
                # # === Log final frame order ===
                # # ================================
                # # Final frame order: sink + history + tail
                # final_frame_indices = torch.cat([sink_frames_before_adjustment,
                #                                 hist_frames_before_adjustment,
                #                                 tail_frames_before_adjustment])
                # # Log with GLOBAL frame indices for verification
                # global_frame_indices = final_frame_indices + frame_offset
                # # Calculate all RoPE frame indices in order: sink + history + tail
                # all_rope_frames = torch.cat([
                #     sink_hist_frames_indices_after_rope_adj,
                #     tail_rope_frames
                # ])
                # with open("final_key_indices.txt", "a") as f:
                #     f.write(f"=== RoPE Position Logging ===\n")
                #     f.write(f"Sink RoPE start_frame: {sink_start_frame if sink_num_frames > 0 else 'N/A'}, frames: {','.join(map(str, sink_rope_frames.cpu().tolist()))}\n")
                #     f.write(f"History RoPE start_frame: {hist_start_frame if num_hist_frames > 0 else 'N/A'}, frames: {','.join(map(str, hist_rope_frames.cpu().tolist()))}\n")
                #     f.write(f"Tail RoPE start_frame: {tail_start_frame}, frames: {','.join(map(str, tail_rope_frames.cpu().tolist()))}\n")
                #     f.write(f"All RoPE frames (sink+hist+tail): {','.join(map(str, all_rope_frames.cpu().tolist()))}\n")
                #     f.write(f"Frame Indices Before Eviction: {','.join(map(str, final_frame_indices.cpu().tolist()))}\n")
                #     f.write(f"Frame Indices After Eviction (Global): {','.join(map(str, global_frame_indices.cpu().tolist()))}\n")
                #     f.write(f"Frame Indices After RoPE Adjustment: {','.join(map(str, all_rope_frames.cpu().tolist()))}\n")
                #     f.write(f"global_end_frame: {global_end_frame}, local_end_index: {local_end_index}, frame_offset: {frame_offset}\n")
                #     f.write("--------------------------------\n")
