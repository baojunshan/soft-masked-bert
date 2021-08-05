import torch
import torch.nn as nn
from transformers import BertModel
import time


class BiGRU(nn.Module):
    def __init__(self, embedding_size, hidden, n_layers, dropout=0.0):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(embedding_size, hidden, num_layers=n_layers,
                          bidirectional=True, dropout=dropout, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        prob = self.sigmoid(self.linear(out))
        return prob


class SoftMaskedBert(nn.Module):
    def __init__(self, pretrained_bert_path, device, mask_token_id, hidden_size=256):
        super(SoftMaskedBert, self).__init__()
        bert = BertModel.from_pretrained(pretrained_bert_path)

        self.config = bert.config
        self.emb_size = bert.config.hidden_size
        self.max_position_embeddings = bert.config.max_position_embeddings
        self.vocab_size = bert.config.vocab_size
        self.device = device
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size

        self.embedding = bert.embeddings.to(device)
        self.detector = BiGRU(self.emb_size, self.hidden_size, n_layers=2)
        self.corrector = bert.encoder.to(device)
        mask_token_id = torch.tensor([[mask_token_id]]).to(device)
        self.mask_e = self.embedding(mask_token_id).detach()
        self.linear = nn.Linear(self.emb_size, self.vocab_size)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, seg_ids, input_mask):
        emb = self.embedding(input_ids=input_ids, token_type_ids=seg_ids)
        err = self.detector(emb)
        soft_emb = err * self.mask_e + (1 - err) * emb
        _, _, _, _, \
        _, \
        head_mask, \
        encoder_hidden_states, \
        encoder_extended_attention_mask = self._init_inputs(input_ids, input_mask)
        h = self.corrector(
            soft_emb,
            attention_mask=encoder_extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask
        )
        h_ = h[0] + emb
        out = self.linear(h_)
        return out, err

    def _init_inputs(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return input_ids, position_ids, token_type_ids, inputs_embeds, \
               extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask


class Trainer:
    def __init__(self, model, device, lr=4e-4, alpha=0.8, epoch=10, steps_per_epoch=None, tokenizer=None, max_length=256, load_path=None):
        self.device = device
        self.lr = lr
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.alpha = alpha
        if load_path is not None:
            model.load_state_dict(torch.load(load_path))
        self.model = model.to(self.device)
        self.ce = nn.CrossEntropyLoss().to(self.device)
        self.bce = nn.BCELoss().to(self.device)

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999)
        )

    def fit(self, data_generator, dev_generator=None):
        self.steps_per_epoch = self.steps_per_epoch or len(data_generator)
        for epoch in range(self.epoch):
            start_time = time.time()
            self.model.train()
            for i, (token_ids, token_seg, token_mask, label_mistake, label_ids) in enumerate(data_generator):
                token_ids = token_ids.to(self.device)
                token_seg = token_seg.to(self.device)
                token_mask = token_mask.to(self.device)
                label_mistake = label_mistake.to(self.device)
                label_ids = label_ids.to(self.device)

                output, err_prob = self.model(input_ids=token_ids, seg_ids=token_seg, input_mask=token_mask)

                correct_loss = self.ce(output.transpose(1, 2), label_ids)
                detect_loss = self.bce(err_prob.squeeze(), label_mistake.float())
                loss = correct_loss * self.alpha + detect_loss * (1 - self.alpha)
                # loss = correct_loss

                # with torch.autograd.set_detect_anomaly(True):
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer.zero_grad()

                print(
                    f"\r[Epoch {epoch + 1:03}/{self.epoch:03}]",
                    f"Batch {i + 1:05}/{self.steps_per_epoch:05}",
                    f"Loss: {loss.item():.5f}",
                    end=""
                )

                if i >= self.steps_per_epoch - 1:
                    break
            print(f"\r" + " " * 70, end="")
            print(
                f"\r[Epoch {epoch + 1}/{self.epoch}]",
                f"Loss {loss.item():5f}",
                f"Time {time.time() - start_time:.2f}"
            )
            if dev_generator is not None:
                self.model.eval()
                n_batch = len(dev_generator)
                total_correct, total_div = 0, 0
                for i, (token_ids, token_seg, token_mask, label_mistake, label_ids) in enumerate(dev_generator):
                    token_ids = token_ids.to(self.device)
                    token_seg = token_seg.to(self.device)
                    token_mask = token_mask.to(self.device)
                    label_mistake = label_mistake.to(self.device)
                    label_ids = label_ids.to(self.device)

                    output, err_prob = self.model(input_ids=token_ids, seg_ids=token_seg, input_mask=token_mask)
                    output = torch.argmax(output, dim=-1)

                    error = output == label_ids
                    # error = torch.cast(error, dtype=torch.)

                    correct = error * label_mistake
                    correct = torch.sum(torch.sum(correct, dim=-1), dim=-1)
                    div = torch.sum(torch.sum(label_mistake, dim=-1), dim=-1)
                    total_div += div.item()
                    total_correct += correct.item()
                    if i >= n_batch - 1:
                        break
                print("Correct ratio from mistake is : ", total_correct / total_div)
            torch.save(self.model.state_dict(), "best_model.bin")

    def inference(self, inputs):
        self.model.eval()
        token_ids = self.tokenizer.encode(text=inputs, max_length=self.max_length)
        token_seg = [0] * len(token_ids)
        token_ids = token_ids[:-1][:self.max_length - 1] + [token_ids[-1]] + [0] * (self.max_length - len(token_ids))
        token_seg = token_seg[:self.max_length] + [0] * (self.max_length - len(token_seg))
        token_mask = [1 if i > 0 else 0 for i in token_ids]
        output, err_prob = self.model(
            input_ids=torch.tensor([token_ids]).to(self.device),
            seg_ids=torch.tensor([token_seg]).to(self.device),
            input_mask=torch.tensor([token_mask]).to(self.device)
        )
        result = torch.argmax(output, dim=-1)[0][1:len(inputs)+1]
        result = self.tokenizer.decode(result)
        return result