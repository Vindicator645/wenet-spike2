#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Decoder self-attention layer definition."""
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's inpu
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.norm3 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)
        else:
            self.concat_linear1 = nn.Identity()
            self.concat_linear2 = nn.Identity()

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (torch.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        # import pdb;pdb.set_trace()
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0]), dim=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(
                self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)[0]), dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(
                self.src_attn(x, memory, memory, memory_mask)[0])
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


class DecoderLayerContext(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        context_attn (torch.nn.Module): Context-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's inpu
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        context_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float = 0.0,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.context_attn = context_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.norm3 = nn.LayerNorm(size, eps=1e-5)
        self.norm4 = nn.LayerNorm(size * 2, eps=1e-5)
        self.norm5 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        bias_hidden: torch.Tensor,
        cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            bias_hidden (torch.Tensor): Encoded context
                (#batch, context_num:batch+1, size)
            cache (torch.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        # import pdb;pdb.set_trace()
        tgt = self.norm1(tgt)
        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]
        
        tgt_hidden = self.src_attn(tgt, memory, memory, memory_mask)[0]
        context_hidden = self.context_attn(tgt, bias_hidden, bias_hidden)[0]

        tgt_hidden = self.norm2(tgt_hidden)
        context_hidden = self.norm3(context_hidden)

        tgt_concat = torch.cat((tgt_hidden,context_hidden), dim=-1)
        residual = tgt_concat
        x = residual + self.dropout(
            self.self_attn(tgt_concat, tgt_concat, tgt_concat, tgt_q_mask)[0])
        
        x = self.norm4(x)
        x = self.dropout(self.feed_forward(x))
        x = self.norm5(x)
        
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


class DecoderLayerContextV2(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        context_attn (torch.nn.Module): Context-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's inpu
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        context_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float = 0.0,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        # self.self_attn = FsmnLayer(input_dim=size*2, out_dim=2*size, hidden_dim=2*size, left_frame=3, right_frame=2,left_dilation=2,right_dilation=1)
        self.self_attn = FsmnLayer(input_dim=size*2, out_dim=2*size, hidden_dim=2*size, left_frame=4, right_frame=0,left_dilation=2,right_dilation=1)
        self.src_attn = src_attn
        self.context_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.norm3 = nn.LayerNorm(size, eps=1e-5)
        self.norm4 = nn.LayerNorm(size * 2, eps=1e-5)
        self.norm5 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        bias_hidden: torch.Tensor,
        cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            bias_hidden (torch.Tensor): Encoded context
                (#batch, context_num:batch+1, size)
            cache (torch.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        # import pdb;pdb.set_trace()
        tgt = self.norm1(tgt)
        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]
        
        tgt_hidden = self.src_attn(tgt, memory, memory, memory_mask)[0]
        context_hidden = self.context_attn(tgt, bias_hidden, bias_hidden)[0]

        tgt_hidden = self.norm2(tgt_hidden)
        context_hidden = self.norm3(context_hidden)

        tgt_concat = torch.cat((tgt_hidden,context_hidden), dim=-1)
        x = self.self_attn(tgt_concat)[0]
        x = self.norm4(x)
        x = self.dropout(self.feed_forward(x))
        x = self.norm5(x)
        
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


class FsmnLayer(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, left_frame=1, right_frame=1, left_dilation=1, right_dilation=1):
        super(FsmnLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.left_dilation = left_dilation
        self.right_dilation = right_dilation
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        if left_frame > 0:
            self.pad_left = nn.ConstantPad1d([left_dilation*left_frame, 0], 0)
            self.conv_left = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=left_frame+1,
                                       dilation=left_dilation, bias=False, groups=hidden_dim)
        if right_frame > 0:
            self.pad_right = nn.ConstantPad1d(
                [-right_dilation, right_dilation*right_frame], 0)
            self.conv_right = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=right_frame,
                                        dilation=right_dilation, bias=False, groups=hidden_dim)
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)
        self.cache_size = left_frame*left_dilation+right_frame*right_dilation+1
        self.cache = torch.zeros([1, self.hidden_dim, self.cache_size])

    def forward(self, x, hidden=None):
        x_data = x.transpose(1, 2)
        p_in = self.conv_in(x_data)
        if self.left_frame > 0:
            p_left = self.pad_left(p_in)
            p_left = self.conv_left(p_left)
        else:
            p_left = 0
        if self.right_frame > 0:
            p_right = self.pad_right(p_in)
            p_right = self.conv_right(p_right)
        else:
            p_right = 0
        p_out = p_in + p_right+p_left
        if hidden is not None:
            p_out = hidden + p_out
        out = F.relu(self.conv_out(p_out))
        out = out.transpose(1,2)
        return out, p_out