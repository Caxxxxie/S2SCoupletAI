import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int, hidden_dim: int, n_layer: int, n_head: int, ff_dim: int, embed_drop: float, hidden_drop: float):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_head, dim_feedforward=ff_dim, dropout=hidden_drop)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def encode(self, x, mask):
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)
        return x

    def forward(self, x, *args):
        # (batch_size, max_seq_len, embed_dim)
        mask = args[0] if len(args) > 0 else None
        tok_emb = self.tok_embedding(x)
        max_seq_len = x.shape[-1]
        pos_emb = self.pos_embedding(torch.arange(max_seq_len).to(x.device))
        x = tok_emb + pos_emb.unsqueeze(0)
        x = self.embed_dropout(x)
        x = self.linear1(x)
        x = self.encode(x, mask)
        x = self.linear2(x)
        probs = torch.matmul(x, self.tok_embedding.weight.t())
        return probs


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=n_layer,
                              dropout=rnn_drop if n_layer > 1 else 0, batch_first=True, bidirectional=True)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def encode(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x, _ = self.bilstm(x)
        return x

    def predict(self, x):
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.t())
        return probs

    def forward(self, x, *args):
        x = self.encode(x)
        return self.predict(x)


class BiLSTMAttn(BiLSTM):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float, n_head: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head)

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        x = x.transpose(0, 1)
        x = self.attn(x, x, x, key_padding_mask=mask)[0].transpose(0, 1)
        return self.predict(x)


class BiLSTMCNN(BiLSTM):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, x, *args):
        x = self.encode(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        return self.predict(x)


class BiLSTMConvAttRes(BiLSTM):
    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int, hidden_dim: int, n_layer: int, embed_drop: float, rnn_drop: float, n_head: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_layer, embed_drop, rnn_drop)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head)
        self.conv = nn.Conv1d(in_channels=hidden_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, *args):
        mask = args[0] if len(args) > 0 else None
        x = self.encode(x)
        res = x
        x = self.conv(x.transpose(1, 2)).relu()
        x = x.permute(2, 0, 1)
        x = self.attn(x, x, x, key_padding_mask=mask)[0].transpose(0, 1)
        x = self.norm(res + x)
        return self.predict(x)


class CNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, embed_drop: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, *args):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2).relu()
        x = self.linear(x)
        probs = torch.matmul(x, self.embedding.weight.t())
        return probs

class Seq2SeqRNN(nn.Module):
    """
    forward(src_ids, src_mask, tgt_in_ids, tgt_mask) -> logits (B, Ttgt, V)
    generate(src_ids, src_mask, bos_id, eos_id, max_len) -> pred_ids (B, <=max_len)
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(embed_drop)

        # encoder: bidirectional
        enc_h = hidden_dim // 2
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=enc_h,
            num_layers=n_layer,
            dropout=rnn_drop if n_layer > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # decoder: unidirectional
        self.decoder = nn.LSTM(
            input_size=embed_dim + hidden_dim,   # hidden_dim = 2*enc_h = ctx dim
            hidden_size=hidden_dim,
            num_layers=n_layer,
            dropout=rnn_drop if n_layer > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # init decoder state from encoder final state
        self.init_h = nn.Linear(hidden_dim, hidden_dim)  # from concat(fwd,bwd) -> dec
        self.init_c = nn.Linear(hidden_dim, hidden_dim)

        # output projection
        self.out_proj = nn.Linear(hidden_dim + hidden_dim, vocab_size)  # [dec_out ; ctx] -> vocab

    def _masked_attention(self, dec_out: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor):
        # dot attention scores: (B, Ttgt, Tsrc)
        scores = torch.bmm(dec_out, enc_out.transpose(1, 2))

        if src_mask is not None:
            # mask PAD positions to -inf
            scores = scores.masked_fill(src_mask.unsqueeze(1), float("-inf"))

        attn_w = F.softmax(scores, dim=-1)  # (B, Ttgt, Tsrc)
        ctx = torch.bmm(attn_w, enc_out)    # (B, Ttgt, H)
        return ctx, attn_w

    def _init_decoder_state(self, enc_hn: torch.Tensor, enc_cn: torch.Tensor):
        """
        enc_hn: (num_layers*2, B, enc_h)
        enc_cn: (num_layers*2, B, enc_h)
        """
        num_layers2, B, enc_h = enc_hn.shape
        L = num_layers2 // 2

        # reshape to (L, 2, B, enc_h)
        h = enc_hn.view(L, 2, B, enc_h)
        c = enc_cn.view(L, 2, B, enc_h)

        # concat directions -> (L, B, 2*enc_h) = (L, B, hidden_dim)
        h_cat = torch.cat([h[:, 0], h[:, 1]], dim=-1)
        c_cat = torch.cat([c[:, 0], c[:, 1]], dim=-1)

        # project to decoder hidden_dim (same as hidden_dim here, but keep linear for flexibility)
        h0 = torch.tanh(self.init_h(h_cat))
        c0 = torch.tanh(self.init_c(c_cat))
        return h0.contiguous(), c0.contiguous()

    def encode(self, src_ids: torch.Tensor):
        """
        src_ids: (B, Tsrc)
        returns enc_out (B,Tsrc,H), (enc_hn, enc_cn)
        """
        x = self.embedding(src_ids)
        x = self.embed_dropout(x)
        enc_out, (enc_hn, enc_cn) = self.encoder(x)
        return enc_out, (enc_hn, enc_cn)

    def forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor,
                tgt_in_ids: torch.Tensor, tgt_mask: torch.Tensor = None):
        enc_out, (enc_hn, enc_cn) = self.encode(src_ids)
        h0, c0 = self._init_decoder_state(enc_hn, enc_cn)

        # decoder token embeddings
        y = self.embedding(tgt_in_ids)
        y = self.embed_dropout(y)

        # first pass: run decoder with zero context to get dec_out, then attend, then run again with ctx
        B, Ttgt, _ = y.shape

        # pass1 with zero ctx
        ctx0 = enc_out.new_zeros((B, Ttgt, self.hidden_dim))
        dec_in1 = torch.cat([y, ctx0], dim=-1)
        dec_out1, _ = self.decoder(dec_in1, (h0, c0))  # (B,Ttgt,H)

        # attention using dec_out1
        ctx, _ = self._masked_attention(dec_out1, enc_out, src_mask)

        # pass2 with ctx
        dec_in2 = torch.cat([y, ctx], dim=-1)
        dec_out2, _ = self.decoder(dec_in2, (h0, c0))

        # final attention ctx (optional, improves a bit)
        ctx2, _ = self._masked_attention(dec_out2, enc_out, src_mask)

        logits = self.out_proj(torch.cat([dec_out2, ctx2], dim=-1))  # (B,Ttgt,V)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        force_len: int,
        comma_id: int = None,
        period_id: int = None,
        comma_en_id: int = None,
        period_en_id: int = None,
    ):
        """
        generate force_len tokens while aligning commas
        """
        self.eval()
        device = src_ids.device
        B, Tsrc = src_ids.shape
        L = int(force_len)
        if L > Tsrc:
            L = Tsrc

        enc_out, (enc_hn, enc_cn) = self.encode(src_ids)
        h, c = self._init_decoder_state(enc_hn, enc_cn)

        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        # helper function to decide if comma is allowed
        def mask_punct_by_alignment(logits_1d: torch.Tensor, punct_id: int, t: int):
            if punct_id is None:
                return logits_1d
            allow = (src_ids[:, t] == punct_id)
            logits_1d[:, punct_id] = logits_1d[:, punct_id].masked_fill(~allow, -1e9)
            return logits_1d

        for t in range(L):
            y_emb = self.embedding(ys[:, -1:])  # (B,1,E)
            y_emb = self.embed_dropout(y_emb)

            # step 1: decoder without ctx
            ctx0 = enc_out.new_zeros((B, 1, self.hidden_dim))
            dec_in1 = torch.cat([y_emb, ctx0], dim=-1)
            dec_out1, _ = self.decoder(dec_in1, (h, c))  # (B,1,H)

            # attention
            scores = torch.bmm(dec_out1, enc_out.transpose(1, 2))  # (B,1,Tsrc)
            if src_mask is not None:
                scores = scores.masked_fill(src_mask.unsqueeze(1), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.bmm(attn, enc_out)  # (B,1,H)

            # step 2: decoder with ctx
            dec_in2 = torch.cat([y_emb, ctx], dim=-1)
            dec_out2, (h, c) = self.decoder(dec_in2, (h, c))  # (B,1,H)

            # final attention (optional but ok)
            scores2 = torch.bmm(dec_out2, enc_out.transpose(1, 2))
            if src_mask is not None:
                scores2 = scores2.masked_fill(src_mask.unsqueeze(1), float("-inf"))
            attn2 = torch.softmax(scores2, dim=-1)
            ctx2 = torch.bmm(attn2, enc_out)

            logits = self.out_proj(torch.cat([dec_out2, ctx2], dim=-1)).squeeze(1)  # (B,V)

            # never generate EOS
            logits[:, eos_id] = -1e9
            logits[:, pad_id] = -1e9

            logits = mask_punct_by_alignment(logits, comma_id, t)
            logits = mask_punct_by_alignment(logits, period_id, t)
            logits = mask_punct_by_alignment(logits, comma_en_id, t)
            logits = mask_punct_by_alignment(logits, period_en_id, t)

            next_id = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)

        # get EOS
        ys = torch.cat([ys, torch.full((B, 1), eos_id, dtype=torch.long, device=device)], dim=1)
        return ys

import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2SeqGRU(nn.Module):
    """
    forward(src_ids, src_mask, tgt_in_ids, tgt_mask) -> logits (B, Ttgt, V)
    generate(src_ids, src_mask, bos_id, eos_id, pad_id, force_len, ...) -> pred_ids (B, <=force_len+2)
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_layer: int, embed_drop: float, rnn_drop: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(embed_drop)

        # encoder: bidirectional, so each direction has hidden_dim//2
        assert hidden_dim % 2 == 0, "hidden_dim must be even for BiGRU encoder"
        enc_h = hidden_dim // 2

        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=enc_h,
            num_layers=n_layer,
            dropout=rnn_drop if n_layer > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # decoder: unidirectional
        # input to decoder at each step = [token_emb ; attention_context]
        self.decoder = nn.GRU(
            input_size=embed_dim + hidden_dim,   # ctx dim = hidden_dim
            hidden_size=hidden_dim,
            num_layers=n_layer,
            dropout=rnn_drop if n_layer > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # init decoder state from encoder final state
        # encoder provides (2 directions) -> concat -> hidden_dim
        self.init_h = nn.Linear(hidden_dim, hidden_dim)

        # output projection
        self.out_proj = nn.Linear(hidden_dim + hidden_dim, vocab_size)  # [dec_out ; ctx] -> vocab

    def _masked_attention(self, dec_out: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor):
        scores = torch.bmm(dec_out, enc_out.transpose(1, 2))  # (B,Ttgt,Tsrc)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask.unsqueeze(1), float("-inf"))
        attn_w = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn_w, enc_out)
        return ctx, attn_w

    def _init_decoder_state(self, enc_hn: torch.Tensor):
        num_layers2, B, enc_h = enc_hn.shape
        L = num_layers2 // 2

        # (L, 2, B, enc_h)
        h = enc_hn.view(L, 2, B, enc_h)

        # concat directions -> (L, B, hidden_dim)
        h_cat = torch.cat([h[:, 0], h[:, 1]], dim=-1)

        # project -> (L, B, hidden_dim)
        h0 = torch.tanh(self.init_h(h_cat))
        return h0.contiguous()

    def encode(self, src_ids: torch.Tensor):
        x = self.embedding(src_ids)
        x = self.embed_dropout(x)
        enc_out, enc_hn = self.encoder(x)
        return enc_out, enc_hn

    def forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor,
                tgt_in_ids: torch.Tensor, tgt_mask: torch.Tensor = None):
        enc_out, enc_hn = self.encode(src_ids)
        h0 = self._init_decoder_state(enc_hn)

        y = self.embedding(tgt_in_ids)
        y = self.embed_dropout(y)

        B, Ttgt, _ = y.shape

        # pass1 with zero ctx
        ctx0 = enc_out.new_zeros((B, Ttgt, self.hidden_dim))
        dec_in1 = torch.cat([y, ctx0], dim=-1)
        dec_out1, _ = self.decoder(dec_in1, h0)  # (B,Ttgt,H)

        # attention using dec_out1
        ctx, _ = self._masked_attention(dec_out1, enc_out, src_mask)

        # pass2 with ctx
        dec_in2 = torch.cat([y, ctx], dim=-1)
        dec_out2, _ = self.decoder(dec_in2, h0)

        # final attention ctx (optional)
        ctx2, _ = self._masked_attention(dec_out2, enc_out, src_mask)

        logits = self.out_proj(torch.cat([dec_out2, ctx2], dim=-1))  # (B,Ttgt,V)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        force_len: int,
        comma_id: int = None,
        period_id: int = None,
        comma_en_id: int = None,
        period_en_id: int = None,
    ):
        """
        generate force_len tokens while aligning punctuation positions
        """
        self.eval()
        device = src_ids.device
        B, Tsrc = src_ids.shape
        L = int(force_len)
        if L > Tsrc:
            L = Tsrc

        enc_out, enc_hn = self.encode(src_ids)
        h = self._init_decoder_state(enc_hn)  # (n_layer,B,H)

        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        def mask_punct_by_alignment(logits_1d: torch.Tensor, punct_id: int, t: int):
            if punct_id is None:
                return logits_1d
            allow = (src_ids[:, t] == punct_id)
            logits_1d[:, punct_id] = logits_1d[:, punct_id].masked_fill(~allow, -1e9)
            return logits_1d

        for t in range(L):
            y_emb = self.embedding(ys[:, -1:])  # (B,1,E)
            y_emb = self.embed_dropout(y_emb)

            # step1: decoder without ctx
            ctx0 = enc_out.new_zeros((B, 1, self.hidden_dim))
            dec_in1 = torch.cat([y_emb, ctx0], dim=-1)
            dec_out1, _ = self.decoder(dec_in1, h)  # (B,1,H)

            # attention
            scores = torch.bmm(dec_out1, enc_out.transpose(1, 2))  # (B,1,Tsrc)
            if src_mask is not None:
                scores = scores.masked_fill(src_mask.unsqueeze(1), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.bmm(attn, enc_out)  # (B,1,H)

            # step2: decoder with ctx
            dec_in2 = torch.cat([y_emb, ctx], dim=-1)
            dec_out2, h = self.decoder(dec_in2, h)  # (B,1,H)

            # final attention (optional)
            scores2 = torch.bmm(dec_out2, enc_out.transpose(1, 2))
            if src_mask is not None:
                scores2 = scores2.masked_fill(src_mask.unsqueeze(1), float("-inf"))
            attn2 = torch.softmax(scores2, dim=-1)
            ctx2 = torch.bmm(attn2, enc_out)

            logits = self.out_proj(torch.cat([dec_out2, ctx2], dim=-1)).squeeze(1)  # (B,V)

            # never generate EOS/PAD inside the forced-length content region
            logits[:, eos_id] = -1e9
            logits[:, pad_id] = -1e9

            logits = mask_punct_by_alignment(logits, comma_id, t)
            logits = mask_punct_by_alignment(logits, period_id, t)
            logits = mask_punct_by_alignment(logits, comma_en_id, t)
            logits = mask_punct_by_alignment(logits, period_en_id, t)

            next_id = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)

        ys = torch.cat([ys, torch.full((B, 1), eos_id, dtype=torch.long, device=device)], dim=1)
        return ys