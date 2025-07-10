import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ConvTransformerEncoder(nn.Module):
    def __init__(self, emb_size, n_heads, n_layers=2, conv_kernel=3, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.conv = nn.Conv1d(
            in_channels=emb_size,
            out_channels=emb_size,
            kernel_size=conv_kernel,
            padding=conv_kernel//2,  # 确保output与input等长
            groups=1
        )
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=4*emb_size,
            activation='gelu',
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, seg: torch.Tensor):
        B, L, E = seg.shape
        seg_cnn = seg.transpose(1,2)   # -> [B, E, L]
        seg_cnn = self.conv(seg_cnn)   # -> [B, E, L], kernel=3
        seg_cnn = self.dropout(seg_cnn)
        seg_cnn = seg_cnn.transpose(1,2)  # -> [B, L, E]
        out = self.encoder(seg_cnn)  # [B, L, E]
        return out


class Encoder(nn.Module):
    def __init__(self,
                 relation_num: int,
                 emb_size: int,
                 device: torch.device,
                 n_heads_pair: int = 4,  # convTransformer for pair
                 n_heads_mem: int = 6,   # final memory-att
                 n_layers_pair: int = 2,
                 dropout: float = 0.1,
                 max_len: int = 8,       # relative pos
                 dynamic_mem_heads: int = 2  # 轻量动态记忆 self-att
    ):
        super().__init__()
        self.relation_num = relation_num
        self.emb_size = emb_size
        self.device = device

        # Embedding
        self.emb = nn.Embedding(relation_num + 1, emb_size, padding_idx=relation_num)

        # Relative position embedding
        self.max_len = max_len
        self.rel_pos = nn.Embedding(self.max_len, emb_size)
        self.dynamic_heads = dynamic_mem_heads
        self.qkv_dynamic = nn.Linear(emb_size, 3*emb_size)
        self.out_dynamic = nn.Linear(emb_size, emb_size)
        self.ln_dynamic  = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

        self.pair_encoder = ConvTransformerEncoder(
            emb_size=emb_size,
            n_heads=n_heads_pair,
            n_layers=n_layers_pair,
            conv_kernel=3,
            dropout=dropout
        )

        # Pair scoring
        self.fc_score = nn.Linear(emb_size, 1)
        self.sigmoid  = nn.Sigmoid()

        # WeightedAverage + final memory-att
        self.fc_q = nn.Linear(emb_size, emb_size)
        self.fc_k = nn.Linear(emb_size, emb_size)
        self.fc_v = nn.Linear(emb_size, emb_size)

        # final multi-head attn
        self.final_mha = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=n_heads_mem,
            dropout=dropout,
            batch_first=True
        )
        self.ln_final1 = nn.LayerNorm(emb_size)
        self.ffn_final = nn.Sequential(
            nn.Linear(emb_size, 4*emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*emb_size, emb_size)
        )
        self.ln_final2 = nn.LayerNorm(emb_size)
        self.register_buffer("gate_mask", torch.ones(relation_num))

    def forward(self, inputs: torch.Tensor):
        """
        inputs: [batch_size, seq_len]  (relation IDs)
        return: pred_head, loss_tensor
        """
        x = self.emb(inputs)  # [B, L, E]
        pos_id = torch.arange(x.size(1), device=self.device)
        pos_id = torch.clamp(pos_id, max=self.max_len-1)
        x = x + self.rel_pos(pos_id)  # [B,L,E]
        relation_mem = self.update_memory()
        B, L, E = x.shape
        seq_list = [x]
        loss_list = []
        step_count = 0
        while step_count < L - 1:
            new_out, step_loss = self.reduce_rel_pairs(seq_list[-1], relation_mem)
            seq_list.append(new_out)
            loss_list.append(step_loss)
            step_count += 1
        final_seq = seq_list[-1]  # [B, <=2, E]
        scores = self.memory_attention(final_seq, relation_mem)  # [B, <=2, |R| + <=2]

        probs = torch.softmax(scores, dim=-1)
        final_loss = Categorical(probs=probs).entropy()   # [B,1]
        loss_list.append(final_loss)

        loss_tensor = torch.cat(loss_list, dim=-1)  # [B, steps]

        pred_head = self.predict_head(scores)       # [B, relation_num+?]
        return pred_head, loss_tensor

    def update_memory(self):
        B = 1  # “批量”只要1, 因为 relation_emb共用
        device = self.device
        rel = torch.arange(self.relation_num, device=device).unsqueeze(0)
        rel_emb = self.emb(rel)  # [1, relation_num, E]
        qkv = self.qkv_dynamic(rel_emb)  # [1, relation_num, 3*E]
        E2 = self.emb_size
        q, k, v = qkv.split(E2, dim=-1)  # each [1, relation_num, E]
        q = q.view(1, self.relation_num, self.dynamic_heads, E2//self.dynamic_heads).transpose(1,2)
        k = k.view(1, self.relation_num, self.dynamic_heads, E2//self.dynamic_heads).transpose(1,2)
        v = v.view(1, self.relation_num, self.dynamic_heads, E2//self.dynamic_heads).transpose(1,2)

        # scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(E2//self.dynamic_heads)
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.matmul(attn, v)   # [1, heads, relation_num, E//heads]

        out = out.transpose(1,2).contiguous().view(1, self.relation_num, E2)
        out = self.out_dynamic(out)  # [1, relation_num, E2]
        out = self.ln_dynamic(rel_emb + self.dropout(out))

        return out.squeeze(0)  # [relation_num, E]

    def reduce_rel_pairs(self, local_seq: torch.Tensor, relation_mem: torch.Tensor):
        B, L, E = local_seq.shape
        if L <= 2:
            return local_seq, torch.zeros((B,1), device=self.device)
        pair_vecs = []
        for i in range(L-1):
            seg = local_seq[:, i:i+2, :]  # [B, 2, E]
            pos_idx = torch.arange(2, device=self.device)
            seg = seg + self.rel_pos(pos_idx)
            enc = self.pair_encoder(seg)  # [B, 2, E]
            hid = enc[:, -1, :]
            pair_vecs.append(hid)
        pair_vecs = torch.stack(pair_vecs, dim=1)  # [B, L-1, E]
        scores = self.sigmoid(self.fc_score(pair_vecs).squeeze(-1))  # [B, L-1]
        idx_max = torch.argmax(scores, dim=-1)  # [B]
        prob = torch.softmax(scores, dim=-1)
        step_loss = Categorical(prob).entropy().unsqueeze(-1) # [B,1]
        full_b = torch.arange(B, device=self.device)
        sel_pair = pair_vecs[full_b, idx_max, :].unsqueeze(1)  # [B,1,E]
        fused = self.memory_attention_fuse(sel_pair, relation_mem) # [B,1,E]
        fused = fused.squeeze(1)
        new_local = local_seq.clone()
        zero = torch.zeros(E, device=self.device)
        new_local[full_b, idx_max, :] = fused
        new_local[full_b, idx_max+1, :] = zero

        new_local = new_local[new_local.sum(dim=-1)!=0]
        new_local = new_local.reshape(B, -1, E)

        return new_local, step_loss

    def memory_attention_fuse(self, pair: torch.Tensor, relation_mem: torch.Tensor):
        B, Lp, E = pair.shape  # Lp=1
        mem = torch.cat((relation_mem.unsqueeze(0), pair), dim=1)  # [1, |R|+1, E] => broadcast -> [B, ...]

        # Q = pair
        q = self.fc_q(pair)
        # K = V = memory
        k = self.fc_k(mem)
        v = self.fc_v(mem)

        attn_out, attn_weight = self.final_mha(q, k, v)  # out: [B,Lp,E], w:[B,Lp,|R|+1]

        x = self.ln_final1(pair + attn_out)
        ffn = self.ffn_final(x)
        x = self.ln_final2(x + ffn)
        return x

    def memory_attention(self, final_seq: torch.Tensor, relation_mem: torch.Tensor):

        B, L, E = final_seq.shape
        mem = torch.cat((relation_mem.unsqueeze(0).expand(B, -1, -1), final_seq), dim=1)
        q = self.fc_q(final_seq)
        k = self.fc_k(mem)
        attn_out, _ = self.final_mha(q, k, k)
        x = self.ln_final1(final_seq + attn_out)
        ffn = self.ffn_final(x)
        x = self.ln_final2(x + ffn)
        logits = torch.matmul(x, mem.transpose(-2,-1)) / math.sqrt(self.emb_size)
        if hasattr(self, "gate_mask"):
            gate = torch.cat([self.gate_mask, torch.ones(L, device=self.device)], dim=0)
            logits = logits * gate

        return logits
    def predict_head(self, scores: torch.Tensor):
        return scores[:, -1, :]

    def get_relation_emb(self, rel: torch.Tensor):
        return self.emb(rel)
