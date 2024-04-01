"""This lobe implements the Stacked Acoustic and Textual Encoders (SATE).
This is best used with pre-trained models.

Currently tesyted and supported models include:
- For text: mBART, mT5
- For speech: wav2vec 2.0, wavLM
Simply used the SpeechBrain HF interfaces to use them.

SATE paper: https://aclanthology.org/2021.acl-long.204/
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Haroun Elleuch 2024
"""

from math import sqrt
from numpy import ndarray
from logging import getLogger

from torch.nn import Module, Sequential, ReLU, Dropout
from torch import (
    FloatTensor,
    LongTensor,
    Tensor,
    mm,
    cat,
    no_grad,
    long,
    ones,
    int64,
)

from speechbrain.nnet.linear import Linear
from speechbrain.nnet.embedding import Embedding
from speechbrain.nnet.activations import Softmax
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.lobes.downsampling import Conv1DDownsampler
from speechbrain.lobes.models.transformer.Transformer import (
    PositionalEncoding,
    get_mask_from_lengths,
)
from speechbrain.lobes.models.huggingface_transformers.mt5 import mT5

logger = getLogger(__name__)


class SATEAdaptor(Module):
    def __init__(
        self,
        acoustic_encoder_embedding_dim: int,
        text_encoder_embedding_dim: int,
        vocab_size: int,
        embed_tokens=None,
        pad_index: int = 0,
        adapter_type: str = "linear",
        scale_embeddings: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_scale = sqrt(acoustic_encoder_embedding_dim)
        if not scale_embeddings:
            self.embed_scale = 1.0

        self.vocab_size = vocab_size
        self.padding_idx = pad_index
        self.dropout_module = Dropout(p=dropout)
        self.adapter_type = adapter_type

        encoder_embedding_dim = acoustic_encoder_embedding_dim

        if adapter_type in [
            "linear",
            "league",
            "gated_league",
            "gated_league2",
        ]:
            self.linear_adapter = Sequential(
                Linear(
                    input_size=acoustic_encoder_embedding_dim,
                    n_neurons=text_encoder_embedding_dim,
                ),
                LayerNorm(input_size=text_encoder_embedding_dim),
                self.dropout_module,
                ReLU(),
            )
        elif adapter_type == "linear2":
            self.linear_adapter = Sequential(
                Linear(
                    acoustic_encoder_embedding_dim, text_encoder_embedding_dim
                ),
                self.dropout_module,
            )
        elif adapter_type == "subsample":
            self.subsample_adaptor = (
                Conv1DDownsampler(  # TODO: implement ! ==> Or skip
                    downsampling_factor=...,
                    kernel_size=...,
                )
            )

        if adapter_type in [
            "embed",
            "context",
            "league",
            "gated_league",
            "gated_league2",
        ]:
            if embed_tokens is None:
                num_embeddings = self.vocab_size
                self.embed_adapter = Embedding(
                    num_embeddings=num_embeddings,
                    embedding_dim=encoder_embedding_dim,
                    blank_id=self.padding_idx,
                )
            else:
                self.embed_adapter = embed_tokens

        if adapter_type == "gated_league":
            self.gate_linear = Linear(
                2 * encoder_embedding_dim, encoder_embedding_dim
            )
        elif adapter_type == "gated_league2":
            self.gate_linear1 = Linear(
                encoder_embedding_dim, encoder_embedding_dim
            )
            self.gate_linear2 = Linear(
                encoder_embedding_dim, encoder_embedding_dim
            )

        self.embed_positions = PositionalEncoding(
            input_size=encoder_embedding_dim,
        )

    # TODO: Refoactor this method into something that adheres better to best practices... ==> Adaptor sub-classes !
    def forward(self, x, padding) -> tuple[Tensor, Tensor, Tensor]:

        representation, distribution = x
        batch, seq_len, embed_dim = distribution.size()
        lengths = (~padding).long().sum(-1)

        if self.adapter_type == "linear":
            out = self.linear_adapter(representation)

        elif self.adapter_type == "context":
            out = mm(
                distribution.view(-1, embed_dim), self.embed_adapter.weight
            ).view(batch, seq_len, -1)

        elif self.adapter_type == "subsample":
            representation = representation.transpose(0, 1)
            out, input_lengths = self.subsample_adaptor(representation, lengths)
            padding = get_mask_from_lengths(input_lengths)

        elif self.adapter_type == "league":
            linear_out = self.linear_adapter(representation)
            soft_out = mm(
                distribution.view(-1, embed_dim), self.embed_adapter.weight
            ).view(batch, seq_len, -1)
            out = linear_out + soft_out
        elif self.adapter_type == "gated_league":
            linear_out = self.linear_adapter(representation)
            soft_out = mm(
                distribution.view(-1, embed_dim), self.embed_adapter.weight
            ).view(batch, seq_len, -1)
            coef = (
                self.gate_linear(cat([linear_out, soft_out], dim=-1))
            ).sigmoid()
            out = coef * linear_out + (1 - coef) * soft_out
        elif self.adapter_type == "none":
            out = representation
        else:
            out = None
            logger.error(f"Unsupported adapter type: {self.adapter_type}.")

        out = self.embed_scale * out

        positions = self.embed_positions(padding).transpose(0, 1)

        out = positions + out

        out = self.dropout_module(out)

        return out, positions, padding


class SATEEncoder(Module):
    """SATE encoder module as described in the paper: ...
    A SATE encode is composed of an acoustic encoder (with a Softmax layer on top), an adpater and a textual encoder.
    The acoustic encoder is also intended to be optimized with a CTC loss objective.

    Arguments
    ----------
    ! TODO: Complete implementation
    """

    def __init__(
        self,
        acoustic_encoder: Module,
        text_encoder: Module,
        vocab_size: int,
        acoustic_encoder_embedding_dim: int = 1024,
        text_encoder_embedding_dim: int = 1024,
        freeze: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.freeze = freeze
        self.acoustic_encoder = acoustic_encoder

        self.adaptor = SATEAdaptor(
            vocab_size=vocab_size,
            acoustic_encoder_embedding_dim=acoustic_encoder_embedding_dim,
            text_encoder_embedding_dim=text_encoder_embedding_dim,
        )

        self.text_encoder = text_encoder
        self.lin_softmax = Sequential(
            Linear(
                input_size=acoustic_encoder_embedding_dim, n_neurons=vocab_size
            ),
            Softmax(apply_log=True),
        )

    def forward(
        self, wav: Tensor | ndarray, wav_lens: Tensor | ndarray | list
    ) -> FloatTensor:
        if self.freeze:
            with no_grad:
                acoustic_features = self.acoustic_encoder.forward(wav)

                logits = self.lin_softmax(acoustic_features)

                encoder_padding_mask = get_mask_from_lengths(wav_lens)
                adaptor_out, _, _ = self.adaptor(
                    x=(acoustic_features, logits), padding=encoder_padding_mask
                )
                encoder_out = self.text_encoder.forward(
                    inputs_embeds=adaptor_out
                )
        else:
            acoustic_features = self.acoustic_encoder.forward(wav)

            logits = self.lin_softmax(acoustic_features)

            encoder_padding_mask = get_mask_from_lengths(wav_lens)
            adaptor_out, _, _ = self.adaptor(
                x=(acoustic_features, logits), padding=encoder_padding_mask
            )
            encoder_out = self.text_encoder.forward(inputs_embeds=adaptor_out)

        return logits, encoder_out.last_hidden_state


class SATE(Module):
    # TODO: Complete docstirings + finish documentation
    # TODO: Do not forget the runnable example (can be derived from the test notebooks)

    def __init__(
        self,
        vocab_size: int,
        acoustic_encoder: Module,
        text_encoder: Module = None,
        text_decoder: Module = None,
        use_seq_to_seq_model_lm_head: bool = False,
        seq_to_seq_text_model: Module = None,
        acoustic_encoder_embedding_dim: int = 1024,
        text_encoder_embedding_dim: int = 1024,
        freeze: bool = False,
        freeze_encoder: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.freeze = freeze

        if (text_encoder or text_decoder) and seq_to_seq_text_model:
            raise ValueError(
                "Either use a sequence-to-sequence text model or give a seaparate text encoder and decoder as arguments."
            )
        elif (
            text_encoder
            and text_decoder is None
            and seq_to_seq_text_model is None
        ):
            raise ValueError("The text decoder cannot be None.")

        elif (
            text_encoder is None
            and text_decoder
            and seq_to_seq_text_model is None
        ):
            raise ValueError("The text encoder cannot be None.")

        if text_encoder and text_decoder:
            self.decoder = text_decoder

        elif seq_to_seq_text_model:
            text_encoder = seq_to_seq_text_model.model.model.encoder
            self.decoder = seq_to_seq_text_model.model.model.decoder

        self._build_sate_encoder(
            acoustic_encoder=acoustic_encoder,
            text_encoder=text_encoder,
            vocab_size=vocab_size,
            acoustic_encoder_embedding_dim=acoustic_encoder_embedding_dim,
            text_encoder_embedding_dim=text_encoder_embedding_dim,
            freeze=self.freeze or freeze_encoder,
        )

        self.lm_head = None
        if use_seq_to_seq_model_lm_head and seq_to_seq_text_model:
            self.lm_head = seq_to_seq_text_model.model.lm_head

    def _build_sate_encoder(
        self,
        acoustic_encoder: Module,
        text_encoder: Module,
        vocab_size: int,
        acoustic_encoder_embedding_dim: int,
        text_encoder_embedding_dim: int,
        freeze: bool,
    ) -> None:
        self.encoder = SATEEncoder(
            acoustic_encoder=acoustic_encoder,
            text_encoder=text_encoder,
            vocab_size=vocab_size,
            acoustic_encoder_embedding_dim=acoustic_encoder_embedding_dim,
            text_encoder_embedding_dim=text_encoder_embedding_dim,
            freeze=freeze,
        )

    def forward_encoder(
        self,
        wav: Tensor | ndarray,
        wav_lens: Tensor | ndarray | list,
    ) -> tuple[Tensor, Tensor]:
        return self.encoder.forward(wav, wav_lens)

    def forward(
        self,
        wav: Tensor | ndarray,
        wav_lens: Tensor | ndarray | list,
        input_ids: LongTensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        acoustic_logits, encoder_representation = self.forward_encoder(
            wav, wav_lens
        )

        if self.freeze:
            with no_grad:
                decoder_output = self.decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=encoder_representation,
                )
                decoder_output = decoder_output.last_hidden_state

                if self.lm_head is not None:
                    decoder_output = self.lm_head(decoder_output)
        else:
            decoder_output = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_representation,
            )
            decoder_output = decoder_output.last_hidden_state

            if self.lm_head is not None:
                decoder_output = self.lm_head(decoder_output)

        return acoustic_logits, encoder_representation, decoder_output

    @no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        enc_len : torch.LongTensor
            The actual length of encoder states.
        """

        if tgt.dtype not in [long, int64]:
            tgt = tgt.long()

        tgt_mask = ones(tgt.size(), device=tgt.device)

        output = self.decoder(
            input_ids=tgt,
            encoder_hidden_states=encoder_out,
            attention_mask=tgt_mask,
            output_attentions=True,
        )

        if self.lm_head:
            return (
                self.lm_head(output.last_hidden_state),
                output.cross_attentions[-1],
            )
        else:
            return (
                output.last_hidden_state,
                output.cross_attentions[-1],
            )


class SATEForMT5(SATE):
    def __init__(
        self,
        vocab_size: int,
        mt5_model: mT5,
        acoustic_encoder: Module,
        use_seq_to_seq_model_lm_head: bool = True,
        embedding_dim: int = 1024,
        freeze: bool = False,
        freeze_encoder: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            acoustic_encoder=acoustic_encoder,
            text_encoder=mt5_model.model.encoder,
            text_decoder=mt5_model.model.decoder,
            seq_to_seq_text_model=None,
            use_seq_to_seq_model_lm_head=use_seq_to_seq_model_lm_head,
            acoustic_encoder_embedding_dim=embedding_dim,
            text_encoder_embedding_dim=embedding_dim,
            freeze=freeze,
            freeze_encoder=freeze_encoder,
            *args,
            **kwargs,
        )

        if use_seq_to_seq_model_lm_head:
            self.lm_head = mt5_model.model.lm_head


class SATELinearAdaptor(SATEAdaptor):
    """A linear SATE adaptor. This is the main adaptor used in the original paper."""

    def __init__(
        self,
        acoustic_encoder_embedding_dim: int,
        text_encoder_embedding_dim: int,
        vocab_size: int,
        embed_tokens=None,
        pad_index: int = 0,
        scale_embeddings: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            acoustic_encoder_embedding_dim,
            text_encoder_embedding_dim,
            vocab_size,
            embed_tokens,
            pad_index,
            scale_embeddings,
            dropout,
        )

    # TODO: Finish implementation.
    # TODO: Implement adaptor freezing.
    # TODO: Add / check needs_gradient argument when freezing modules.

    def forward(self, x, padding) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError()
