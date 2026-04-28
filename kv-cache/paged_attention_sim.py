"""
Simulated Paged Attention for KV-Cache Layout Determinism Tests.

Implements a simple paged KV-cache manager and paged attention kernel
in pure PyTorch, allowing precise control over physical page layout.
"""

import torch
import math
from dataclasses import dataclass


@dataclass
class PagedKVCache:
    """Paged KV-Cache with controllable physical layout."""
    num_layers: int
    num_heads: int
    head_dim: int
    page_size: int
    max_pages: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        # Physical page pool: [max_pages, 2(k+v), num_heads, page_size, head_dim]
        self.page_pool = torch.zeros(
            self.max_pages, 2, self.num_heads, self.page_size, self.head_dim,
            dtype=self.dtype, device=self.device,
        )
        # Track which pages are free
        self.free_pages = list(range(self.max_pages))
        # Per-layer page tables: layer_idx -> list of page indices
        self.page_tables = {i: [] for i in range(self.num_layers)}
        # Tokens stored per layer
        self.seq_lens = {i: 0 for i in range(self.num_layers)}

    def reset(self):
        self.page_pool.zero_()
        self.free_pages = list(range(self.max_pages))
        self.page_tables = {i: [] for i in range(self.num_layers)}
        self.seq_lens = {i: 0 for i in range(self.num_layers)}

    def allocate_pages(self, n_pages: int, allocation_order: list[int] | None = None) -> list[int]:
        """Allocate pages from pool. allocation_order controls which physical pages to use."""
        if allocation_order is not None:
            pages = allocation_order[:n_pages]
            for p in pages:
                if p in self.free_pages:
                    self.free_pages.remove(p)
            return pages
        else:
            pages = [self.free_pages.pop(0) for _ in range(n_pages)]
            return pages

    def append_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
                  allocation_order: list[int] | None = None):
        """
        Append KV tensors to cache for a given layer.
        k, v: [num_heads, new_tokens, head_dim]
        """
        new_tokens = k.shape[1]
        current_len = self.seq_lens[layer_idx]
        current_pages = self.page_tables[layer_idx]

        # How many tokens fit in current last page
        tokens_in_last_page = current_len % self.page_size if current_pages else 0
        remaining_in_last = self.page_size - tokens_in_last_page if tokens_in_last_page > 0 else 0

        written = 0
        # Fill remaining slots in last page
        if remaining_in_last > 0 and current_pages:
            n = min(remaining_in_last, new_tokens)
            page_idx = current_pages[-1]
            start = tokens_in_last_page
            self.page_pool[page_idx, 0, :, start:start+n, :] = k[:, written:written+n, :]
            self.page_pool[page_idx, 1, :, start:start+n, :] = v[:, written:written+n, :]
            written += n

        # Allocate new pages for remaining tokens
        tokens_left = new_tokens - written
        if tokens_left > 0:
            n_new_pages = math.ceil(tokens_left / self.page_size)
            new_pages = self.allocate_pages(n_new_pages, allocation_order)
            for page_idx in new_pages:
                n = min(self.page_size, tokens_left)
                self.page_pool[page_idx, 0, :, :n, :] = k[:, written:written+n, :]
                self.page_pool[page_idx, 1, :, :n, :] = v[:, written:written+n, :]
                written += n
                tokens_left -= n
            current_pages.extend(new_pages)

        self.seq_lens[layer_idx] = current_len + new_tokens

    def get_kv_contiguous(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read KV back as contiguous tensors by walking page table in order."""
        seq_len = self.seq_lens[layer_idx]
        pages = self.page_tables[layer_idx]
        k_out = torch.zeros(self.num_heads, seq_len, self.head_dim,
                            dtype=self.dtype, device=self.device)
        v_out = torch.zeros(self.num_heads, seq_len, self.head_dim,
                            dtype=self.dtype, device=self.device)
        pos = 0
        for i, page_idx in enumerate(pages):
            n = min(self.page_size, seq_len - pos)
            k_out[:, pos:pos+n, :] = self.page_pool[page_idx, 0, :, :n, :]
            v_out[:, pos:pos+n, :] = self.page_pool[page_idx, 1, :, :n, :]
            pos += n
        return k_out, v_out


def paged_attention(
    q: torch.Tensor,
    page_pool: torch.Tensor,
    page_table: list[int],
    seq_len: int,
    page_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Paged attention: compute attention by iterating over pages.
    This simulates how real paged attention kernels do partial reduction per page.

    q: [num_heads, q_len, head_dim]
    page_pool: [max_pages, 2, num_heads, page_size, head_dim]
    page_table: list of page indices for this sequence
    seq_len: total KV sequence length
    page_size: tokens per page

    Returns: [num_heads, q_len, head_dim]
    """
    num_heads, q_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Online softmax attention over pages (numerically stable)
    # m: running max, l: running sum of exp, acc: running weighted sum
    m = torch.full((num_heads, q_len), float('-inf'), dtype=torch.float32, device=q.device)
    l = torch.zeros(num_heads, q_len, dtype=torch.float32, device=q.device)
    acc = torch.zeros(num_heads, q_len, head_dim, dtype=torch.float32, device=q.device)

    pos = 0
    for page_idx in page_table:
        n = min(page_size, seq_len - pos)
        if n <= 0:
            break

        k_page = page_pool[page_idx, 0, :, :n, :]  # [num_heads, n, head_dim]
        v_page = page_pool[page_idx, 1, :, :n, :]

        # Compute attention scores for this page
        # q: [num_heads, q_len, head_dim], k_page: [num_heads, n, head_dim]
        scores = torch.bmm(q.float(), k_page.float().transpose(1, 2)) * scale  # [num_heads, q_len, n]

        # Online softmax merge
        page_max = scores.max(dim=-1).values  # [num_heads, q_len]
        new_m = torch.maximum(m, page_max)

        # Rescale old accumulator
        old_scale = torch.exp(m - new_m)
        # New page contribution
        page_exp = torch.exp(scores - new_m.unsqueeze(-1))  # [num_heads, q_len, n]
        page_sum = page_exp.sum(dim=-1)  # [num_heads, q_len]

        acc = acc * old_scale.unsqueeze(-1) + torch.bmm(page_exp, v_page.float())
        l = l * old_scale + page_sum
        m = new_m

        pos += n

    # Normalize
    out = acc / l.unsqueeze(-1)
    return out.to(q.dtype)


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Standard scaled dot-product attention on contiguous KV."""
    num_heads, q_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    scores = torch.bmm(q.float(), k.float().transpose(1, 2)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.bmm(attn, v.float())
    return out.to(q.dtype)
