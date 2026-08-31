# Adapted from Deep-Forcing under Apache-2.0.
# See third_party/deep_forcing/LICENSE and THIRD_PARTY_NOTICES.md.
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

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

def _rope_time_delta_mul_(k_chunk: torch.Tensor, freqs: torch.Tensor, delta_frames: int) -> None:
    """
    k_chunk: [B, L_sink, H, D] (view of the sink portion of K, with RoPE already applied)
    freqs  : [1024, C/2] complex (self.freqs)
    delta_frames: how many frames to shift the sink to the left (negative) / right (positive)
    In-place, multiplies only the time-axis channels by exp(i * ω * delta_frames)
    """
    if delta_frames == 0:
        return

    B, L, H, D = k_chunk.shape
    assert D % 2 == 0
    c = D // 2
    t_c = c - 2 * (c // 3)   # time channel complex dim
    h_c = c // 3
    w_c = c // 3
    # freqs -> time / height / width split
    freqs_t, _, _ = freqs.split([t_c, h_c, w_c], dim=1)  # [1024, t_c], complex

    #  Complex rotation factor corresponding to the delta (for the time axis)
    shift = abs(int(delta_frames))
    max_pos = freqs_t.shape[0] - 1
    if shift > max_pos:
        shift = max_pos
    mult = freqs_t[shift] if delta_frames >= 0 else torch.conj(freqs_t[shift])
    mult = mult.view(1, 1, 1, t_c)  # [1,1,1,t_c]

    # Convert only the time-axis channels to complex and multiply (in-place)
    time_ri = k_chunk[..., : 2 * t_c]                                            # [B,L,H,2*t_c]
    time_cx = torch.view_as_complex(time_ri.to(torch.float64).reshape(-1, t_c, 2))  # [(B*L*H), t_c]
    time_cx = time_cx * mult.to(time_cx.dtype)                                   # delta rotate
    time_ri_new = torch.view_as_real(time_cx).reshape(B, L, H, t_c, 2).flatten(-2)  # [B,L,H,2*t_c]
    time_ri.copy_(time_ri_new.to(time_ri.dtype))  # in-place

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

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

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)



class PCConfig:
    def __init__(self, enable=False, capacity=1560*15, window=1560*4, fusion="sum", keep_sinks=True, topc_max_reuse=0):
        """
        Args:
            enable (bool): turn PC on/off (in inference path with kv_cache)
            capacity (int): total KV capacity (C + R [+ sink])
            window (int): R (recent window size)
            fusion (str): 'sum' or 'max'  (fuses recent attention scores)
            keep_sinks (bool): always keep sink region (first sink_tokens) if any
            topc_max_reuse (int): max number of Top-C selections allowed per token before it is excluded (0 disables limit)
        """
        self.enable = enable
        self.capacity = int(capacity)
        self.window = int(window)
        self.fusion = fusion
        self.keep_sinks = keep_sinks
        self.topc_max_reuse = max(0, int(topc_max_reuse))

def _mkv_update_win_q(kv_cache, new_win_q, R):
    """Append new recent queries and keep only last R."""
    new_win_q = new_win_q.detach()
    if "win_q" not in kv_cache or kv_cache["win_q"] is None:
        kv_cache["win_q"] = new_win_q
    else:
        kv_cache["win_q"] = torch.cat([kv_cache["win_q"], new_win_q], dim=1)
    # trim to last R
    if kv_cache["win_q"].shape[1] > R:
        kv_cache["win_q"] = kv_cache["win_q"][:, -R:]

def _mkv_select_indices(scores_fused, T, R_eff, sink_tokens, top_c, device, topc_counts=None, topc_max_reuse=0):
    """
    Build keep indices per-batch:
      keep = [sink indices (optional)] + top_c_from_old + [recent R indices]
    scores_fused: [B, T] fused scores over heads & recent window
    """
    B = scores_fused.shape[0]
    keep_lists = []
    protect_lists = []
    topc_selected_lists = []
    recent_idx = torch.arange(max(0, T - R_eff), T, device=device)  # [R_eff]
    sink_idx = torch.arange(0, sink_tokens, device=device) if sink_tokens > 0 else torch.tensor([], device=device, dtype=torch.long)

    # candidate old region = [sink_tokens, T - R_eff)
    old_end = max(0, T - R_eff)
    cand_start = min(sink_tokens, old_end)  # guard
    cand_len = max(0, old_end - cand_start)
    if cand_len == 0 or top_c <= 0:
        # only sinks + recents
        for _ in range(B):
            keep_lists.append(torch.cat([sink_idx, recent_idx]).clone())
            protect_lists.append(torch.tensor([], device=device, dtype=torch.long))
            topc_selected_lists.append(torch.tensor([], device=device, dtype=torch.long))
        return keep_lists, protect_lists, topc_selected_lists

    candidate_idx = torch.arange(cand_start, old_end, device=device)
    k = min(int(top_c), cand_len)

    for b in range(B):
        scores_b = scores_fused[b, cand_start:old_end]
        counts_b = None
        if topc_counts is not None:
            counts_b = topc_counts[b, cand_start:old_end]
        if counts_b is not None and topc_max_reuse > 0:
            valid_mask = counts_b < topc_max_reuse
            if not torch.any(valid_mask):
                selected_b = torch.tensor([], device=device, dtype=torch.long)
            else:
                allowed_scores = scores_b[valid_mask]
                allowed_idx = candidate_idx[valid_mask]
                k_eff = min(k, allowed_idx.numel())
                if k_eff > 0:
                    _, top_local = torch.topk(allowed_scores, k=k_eff, dim=0)
                    selected_b = torch.sort(allowed_idx[top_local])[0]
                else:
                    selected_b = torch.tensor([], device=device, dtype=torch.long)
        else:
            k_eff = min(k, candidate_idx.numel())
            if k_eff > 0:
                _, top_local = torch.topk(scores_b, k=k_eff, dim=0)
                selected_b = torch.sort(candidate_idx[top_local])[0]
            else:
                selected_b = torch.tensor([], device=device, dtype=torch.long)

        protect_b = torch.unique(selected_b, sorted=True)
        keep_b = torch.unique(torch.cat([sink_idx, protect_b, recent_idx]), sorted=True)
        keep_lists.append(keep_b)
        protect_lists.append(protect_b)
        topc_selected_lists.append(protect_b)
    return keep_lists, protect_lists, topc_selected_lists

def _mkv_prune_cache(
    kv_cache,
    keep_lists,
    protect_lists,
    sink_tokens,
    topc_selected_lists=None,
    topc_max_reuse=0,
    source_k=None,
    source_v=None,
    source_abs=None,
    source_topc_counts=None,
):
    """
    In-place-like prune: move kept positions to the front of the preallocated buffer.
    kv_cache['k'/'v']: [B, T, H, D]
    """
    K_dst = kv_cache["k"]; V_dst = kv_cache["v"]
    K_src = source_k if source_k is not None else K_dst
    V_src = source_v if source_v is not None else V_dst
    B, T, H, D = K_dst.shape
    device = K_dst.device
    max_keep = max([len(idx) for idx in keep_lists]) if keep_lists else 0

    # gather per batch
    newK = torch.zeros((B, max_keep, H, D), dtype=K_dst.dtype, device=device)
    newV = torch.zeros((B, max_keep, H, D), dtype=V_dst.dtype, device=device)
    newMask = torch.zeros((B, max_keep), dtype=torch.bool, device=device)
    abs_idx = kv_cache.get("abs_frame_idx", None)
    abs_src = source_abs if source_abs is not None else abs_idx
    topc_counts = kv_cache.get("topc_select_counts", None)
    counts_src = source_topc_counts if source_topc_counts is not None else topc_counts
    newCounts = None
    if topc_counts is not None:
        newCounts = torch.zeros_like(topc_counts)
    if abs_idx is not None:
        newAbs = torch.full((B, max_keep), -1, dtype=abs_idx.dtype, device=device)
    for b in range(B):
        idx = keep_lists[b]
        keep_len = len(idx)
        if keep_len == 0:
            continue
        gather_index = idx.view(-1, 1, 1).expand(keep_len, H, D)  # [keep, H, D]
        newK[b, :keep_len] = torch.gather(K_src[b], dim=0, index=gather_index)
        newV[b, :keep_len] = torch.gather(V_src[b], dim=0, index=gather_index)
        protect_idx = protect_lists[b]
        if protect_idx.numel() > 0:
            positions = torch.searchsorted(idx, protect_idx)
            newMask[b, positions] = True
        if abs_idx is not None and abs_src is not None:
            newAbs[b, :keep_len] = abs_src[b, idx]
        if newCounts is not None and keep_len > 0 and counts_src is not None:
            newCounts[b, :keep_len] = counts_src[b, idx]
            if topc_selected_lists is not None and topc_max_reuse > 0:
                selected = topc_selected_lists[b]
                if selected.numel() > 0:
                    sel_positions = torch.searchsorted(idx, selected)
                    newCounts[b, sel_positions] += 1

    # write back to the front; zero the rest to avoid stale values
    K_dst[:, :max_keep] = newK
    V_dst[:, :max_keep] = newV
    if T > max_keep:
        K_dst[:, max_keep:] = 0
        V_dst[:, max_keep:] = 0
    if abs_idx is not None:
        abs_idx[:, :max_keep] = newAbs
        if T > max_keep:
            abs_idx[:, max_keep:] = -1
    if newCounts is not None:
        topc_counts.zero_()
        topc_counts[:, :max_keep] = newCounts[:, :max_keep]

    # update indices and protection metadata
    kv_cache["local_end_index"].fill_(max_keep)
    cache_capacity = K_dst.shape[1]
    kv_cache["protected_mask"] = torch.zeros((B, cache_capacity), dtype=torch.bool, device=device)
    kv_cache["protected_mask"][:, :max_keep] = newMask
    kv_cache["protected_len"] = kv_cache["protected_mask"].sum(dim=1).to(torch.long)
    kv_cache["protected_len_max"] = kv_cache["protected_len"].max() if kv_cache["protected_len"].numel() > 0 else torch.tensor(0, dtype=torch.long, device=device)

class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 PC: PCConfig | None = None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        self.PC = PC or PCConfig(enable=False)

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
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

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            sink_tokens_v = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            prev_local_end = kv_cache["local_end_index"].item()
            B_cache = kv_cache["k"].shape[0]
            if "abs_frame_idx" not in kv_cache:
                kv_cache["abs_frame_idx"] = torch.full(
                    (kv_cache["k"].shape[0], kv_cache_size),
                    -1,
                    dtype=torch.long,
                    device=kv_cache["k"].device,
                )
            abs_frame_idx = kv_cache["abs_frame_idx"]
            if "topc_select_counts" not in kv_cache:
                kv_cache["topc_select_counts"] = torch.zeros(
                    (kv_cache["k"].shape[0], kv_cache_size),
                    dtype=torch.long,
                    device=kv_cache["k"].device,
                )
            topc_select_counts = kv_cache["topc_select_counts"]
            if abs_frame_idx is not None:
                token_offsets_new = torch.arange(num_new_tokens, device=kv_cache["k"].device)
                new_abs_frames = current_start_frame + torch.div(token_offsets_new, frame_seqlen, rounding_mode='floor')
                new_abs_frames = new_abs_frames.to(abs_frame_idx.dtype)
            else:
                new_abs_frames = None
            rolled_condition = self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + prev_local_end > kv_cache_size)
            available_slots = kv_cache_size - prev_local_end
            need_evict = False
            new_tokens_integrated = False

            if self.PC.enable:
                _mkv_update_win_q(kv_cache, roped_query, R=self.PC.window)

            if rolled_condition:
                if self.PC.enable and available_slots < num_new_tokens:
                    cached_len = prev_local_end
                    K_existing = kv_cache["k"][:, :cached_len]
                    V_existing = kv_cache["v"][:, :cached_len]
                    if cached_len > 0:
                        K_aug = torch.cat([K_existing, roped_key], dim=1)
                        V_aug = torch.cat([V_existing, v], dim=1)
                        if abs_frame_idx is not None and new_abs_frames is not None:
                            new_abs_tile = new_abs_frames.unsqueeze(0).expand(B_cache, -1)
                            abs_existing = abs_frame_idx[:, :cached_len]
                            abs_aug = torch.cat([abs_existing, new_abs_tile], dim=1)
                        else:
                            abs_aug = None
                        if topc_select_counts is not None:
                            zeros_new = torch.zeros((B_cache, num_new_tokens), dtype=topc_select_counts.dtype, device=topc_select_counts.device)
                            counts_existing = topc_select_counts[:, :cached_len]
                            counts_aug = torch.cat([counts_existing, zeros_new], dim=1)
                        else:
                            counts_aug = None
                    else:
                        K_aug = roped_key
                        V_aug = v
                        abs_aug = new_abs_frames.unsqueeze(0).expand(B_cache, -1) if (abs_frame_idx is not None and new_abs_frames is not None) else None
                        counts_aug = torch.zeros((B_cache, num_new_tokens), dtype=topc_select_counts.dtype, device=topc_select_counts.device) if topc_select_counts is not None else None
                    cached_len_aug = cached_len + num_new_tokens
                    win_q = kv_cache.get("win_q", None)
                    R_eff = min(win_q.shape[1], cached_len_aug) if win_q is not None else 0
                    if R_eff > 0 and cached_len_aug > 0:
                        recent_q = win_q[:, -R_eff:]
                        k_flat = K_aug.reshape(K_aug.shape[0], cached_len_aug, -1)
                        k_trans = k_flat.transpose(1, 2).contiguous()
                        scale = 1.0 / (math.sqrt(self.head_dim) * self.num_heads)
                        if self.PC.fusion == "sum":
                            q_sum_flat = recent_q.reshape(recent_q.shape[0], R_eff, -1).sum(dim=1, keepdim=True)
                            fused = torch.bmm(q_sum_flat, k_trans).squeeze(1).to(torch.float32)
                            fused.mul_(scale)
                        else:
                            q_flat = recent_q.reshape(recent_q.shape[0], R_eff, -1)
                            fused = torch.full(
                                (q_flat.shape[0], cached_len_aug),
                                -float("inf"),
                                device=K_aug.device,
                                dtype=torch.float32
                            )
                            step = max(1, min(256, R_eff))
                            for start in range(0, R_eff, step):
                                end = min(R_eff, start + step)
                                q_chunk = q_flat[:, start:end]
                                scores_chunk = torch.matmul(q_chunk, k_trans).to(torch.float32)
                                scores_chunk.mul_(scale)
                                chunk_max = scores_chunk.amax(dim=1)
                                fused = torch.maximum(fused, chunk_max)
                        if self.PC.keep_sinks:
                            forced_sink_tokens = max(sink_tokens, sink_tokens_v)
                            forced_sink = min(forced_sink_tokens, cached_len_aug)
                        else:
                            forced_sink = 0
                        fused[:, max(0, cached_len_aug - R_eff):] = -float("inf")
                        if forced_sink > 0:
                            fused[:, :forced_sink] = -float("inf")
                        total_cap = min(int(self.PC.capacity), kv_cache_size)
                        total_cap = max(total_cap, forced_sink + R_eff)
                        total_cap = min(total_cap, cached_len_aug)
                        top_c = max(0, total_cap - forced_sink - R_eff)
                        keep_lists, protect_lists, topc_selected = _mkv_select_indices(
                            fused, T=cached_len_aug, R_eff=R_eff,
                            sink_tokens=forced_sink, top_c=top_c, device=K_aug.device,
                            topc_counts=counts_aug,
                            topc_max_reuse=self.PC.topc_max_reuse,
                        )
                        _mkv_prune_cache(
                            kv_cache,
                            keep_lists,
                            protect_lists,
                            forced_sink,
                            topc_selected_lists=topc_selected,
                            topc_max_reuse=self.PC.topc_max_reuse,
                            source_k=K_aug,
                            source_v=V_aug,
                            source_abs=abs_aug,
                            source_topc_counts=counts_aug,
                        )
                        prev_local_end = kv_cache["local_end_index"].item()
                        available_slots = kv_cache_size - prev_local_end
                        new_tokens_integrated = True
                if not new_tokens_integrated and available_slots < num_new_tokens:
                    need_evict = True

            if need_evict:
                num_evicted_tokens = num_new_tokens + prev_local_end - kv_cache_size
                num_rolled_tokens = prev_local_end - num_evicted_tokens - sink_tokens
                num_rolled_tokens_v = prev_local_end - num_evicted_tokens - sink_tokens_v
                num_rolled_tokens = max(0, num_rolled_tokens)
                num_rolled_tokens_v = max(0, num_rolled_tokens_v)
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                )
                kv_cache["v"][:, sink_tokens_v:sink_tokens_v + num_rolled_tokens_v] = (
                    kv_cache["v"][:, sink_tokens_v + num_evicted_tokens:sink_tokens_v + num_evicted_tokens + num_rolled_tokens_v].clone()
                )
                abs_frame_idx[:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                    abs_frame_idx[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens]
                )
                if sink_tokens + num_rolled_tokens < prev_local_end:
                    abs_frame_idx[:, sink_tokens + num_rolled_tokens:prev_local_end] = -1
                local_end_index = prev_local_end + current_end - kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
                topc_select_counts[:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                    topc_select_counts[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                )
                if sink_tokens + num_rolled_tokens < prev_local_end:
                    topc_select_counts[:, sink_tokens + num_rolled_tokens:prev_local_end] = 0
                topc_select_counts[:, local_start_index:local_end_index] = 0
            elif not new_tokens_integrated:
                local_end_index = prev_local_end + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
                topc_select_counts[:, local_start_index:local_end_index] = 0
            else:
                local_end_index = kv_cache["local_end_index"].item()
                local_start_index = max(0, local_end_index - num_new_tokens)

            if not new_tokens_integrated:
                insert_len = local_end_index - local_start_index
                if insert_len > 0:
                    token_offsets = torch.arange(insert_len, device=kv_cache["k"].device)
                    abs_frames = current_start_frame + torch.div(token_offsets, frame_seqlen, rounding_mode='floor')
                    abs_frame_idx[:, local_start_index:local_end_index] = abs_frames
            elif abs_frame_idx is not None and kv_cache["k"].shape[1] > local_end_index:
                abs_frame_idx[:, local_end_index:] = -1

            window_start = max(0, local_end_index - self.max_attention_size)
            key_win = kv_cache["k"][:, window_start:local_end_index]
            val_win = kv_cache["v"][:, window_start:local_end_index]
            if rolled_condition:
                sink_len_tokens = min(local_end_index, sink_tokens)

                # tail (recent) range
                tail_end   = local_end_index
                tail_start = max(sink_tokens, local_end_index - self.max_attention_size + sink_tokens)

                # Calculate absolute frames
                tail_len_tokens = tail_end - tail_start
                tail_len_frames = tail_len_tokens // frame_seqlen
                sink_len_frames = sink_len_tokens // frame_seqlen

                tail_start_abs_frame = current_start_frame - tail_len_frames
                desired_sink_abs_start = tail_start_abs_frame - sink_len_frames

                # sink delta rotation (in-place) — only time-axis channels
                if self.sink_size > 0 and sink_len_tokens > 0:
                    if "sink_base_abs_start_frame" not in kv_cache:
                        kv_cache["sink_base_abs_start_frame"] = torch.tensor(desired_sink_abs_start, device=kv_cache["k"].device)
                        delta = 21 - (self.PC.capacity // 1560) ## 18: 3, 17:4, 16:5, 15:6
                    else:
                        delta = int(desired_sink_abs_start - kv_cache["sink_base_abs_start_frame"].item())

                    if delta != 0:
                        _rope_time_delta_mul_(kv_cache["k"][:, :sink_len_tokens], freqs, delta)
                        # 기준 업데이트
                        kv_cache["sink_base_abs_start_frame"].fill_(desired_sink_abs_start)
                    if "abs_frame_idx" in kv_cache and sink_len_tokens > 0:
                        sink_offsets = torch.arange(sink_len_tokens, device=kv_cache["k"].device)
                        sink_frames = desired_sink_abs_start + torch.div(sink_offsets, frame_seqlen, rounding_mode='floor')
                        kv_cache["abs_frame_idx"][:, :sink_len_tokens] = sink_frames

                    key_win = torch.cat([
                        kv_cache["k"][:, :sink_len_tokens],
                        kv_cache["k"][:, tail_start:tail_end]
                    ], dim=1)
                    val_win = torch.cat([
                        kv_cache["v"][:, :sink_len_tokens],
                        kv_cache["v"][:, tail_start:tail_end]
                    ], dim=1)
                else:
                    key_win = kv_cache["k"][:, tail_start:tail_end]
                    val_win = kv_cache["v"][:, tail_start:tail_end]

                # Top-C RoPE compensation (after sink, before tail)
                top_start = sink_tokens
                top_end = tail_start
                top_len_tokens = max(0, top_end - top_start)
                if top_len_tokens > 0:
                    top_len_frames = math.ceil(top_len_tokens / frame_seqlen)
                    desired_top_abs_start = tail_start_abs_frame - top_len_frames
                    if "topc_base_abs_start_frame" not in kv_cache:
                        kv_cache["topc_base_abs_start_frame"] = torch.tensor(desired_top_abs_start, device=kv_cache["k"].device)
                    delta_top = int(desired_top_abs_start - kv_cache["topc_base_abs_start_frame"].item())
                    if delta_top != 0:
                        _rope_time_delta_mul_(kv_cache["k"][:, top_start:top_end], freqs, delta_top)
                        kv_cache["topc_base_abs_start_frame"].fill_(desired_top_abs_start)
                    if "abs_frame_idx" in kv_cache:
                        top_offsets = torch.arange(top_len_tokens, device=kv_cache["k"].device)
                        top_frames = desired_top_abs_start + torch.div(top_offsets, frame_seqlen, rounding_mode='floor')
                        kv_cache["abs_frame_idx"][:, top_start:top_end] = top_frames

            x = attention(roped_query, key_win, val_win)

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 PC: PCConfig | None = None):
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
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps, PC=PC)
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
        cache_start=None
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
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

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
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 PC_enable: bool = True,
                 PC_capacity: int = 1560*16,
                 PC_window: int = 1560 * 4,
                 PC_fusion: str = "sum",
                 PC_keep_sinks: bool = True,
                 PC_topc_max_reuse: int = 7,):
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
            PC_topc_max_reuse (`int`, *optional*, defaults to 0):
                Maximum number of Top-C selections per token before the token is excluded from Top-C (0 disables).
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
        self.PC_cfg = PCConfig(
            enable=PC_enable,
            capacity=PC_capacity,
            window=PC_window,
            fusion=PC_fusion,
            keep_sinks=PC_keep_sinks,
            topc_max_reuse=PC_topc_max_reuse,
        )
        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                PC=self.PC_cfg
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

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

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

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

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

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

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
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
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
