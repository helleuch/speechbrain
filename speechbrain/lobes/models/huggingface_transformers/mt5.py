"""This lobe enables the integration of huggingface pretrained mT5 models.


Authors
 * Haroun Elleuch, 2024 
"""

import torch
import logging

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)

logger = logging.getLogger(__name__)


class mT5(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained mBART models.
    TODO: Complete the documentation here
    """

    def __init__(
        self,
        source,
        save_path,
        freeze=True,
        decoder_only=False,
    ):
        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            seq2seqlm=True,
        )

        self.decoder_only = decoder_only

        self.load_tokenizer(source=source)

        if decoder_only:
            # When we only want to use the decoder part
            del self.model.model.encoder

    def forward(self, src, tgt, pad_idx=0):
        """This method implements a forward step for mt task ...
        Arguments
        ----------
        src (transcription): tensor
            output features from the w2v2 encoder
        tgt (translation): tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """

        tgt = self.custom_padding(
            tgt, 0, self.model.model.decoder.config.pad_token_id
        )

        if self.freeze:
            with torch.no_grad():
                if hasattr(self.model.model, "encoder"):
                    src = self.model.model.encoder(
                        inputs_embeds=src
                    ).last_hidden_state.detach()
                dec_out = self.model.model.decoder(
                    input_ids=tgt, encoder_hidden_states=src
                ).last_hidden_state.detach()
                dec_out = self.model.lm_head(dec_out).detach()
                return dec_out

        if hasattr(self.model.model, "encoder"):
            src = self.model.model.encoder(inputs_embeds=src).last_hidden_state
        dec_out = self.model.model.decoder(
            input_ids=tgt, encoder_hidden_states=src
        ).last_hidden_state
        dec_out = self.model.lm_head(dec_out)
        return dec_out

    @torch.no_grad()
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

        if tgt.dtype not in [torch.long, torch.int64]:
            tgt = tgt.long()

        tgt_mask = torch.ones(tgt.size(), device=tgt.device)

        output = self.model.model.decoder(
            input_ids=tgt,
            encoder_hidden_states=encoder_out,
            attention_mask=tgt_mask,
            output_attentions=True,
        )

        return (
            self.model.lm_head(output.last_hidden_state),
            output.cross_attentions[-1],
        )

    def custom_padding(self, x, org_pad, custom_pad):
        """This method customizes the padding.
        Default pad_idx of SpeechBrain is 0.
        However, it happens that some text-based models like mBART reserves 0 for something else,
        and are trained with specific pad_idx.
        This method change org_pad to custom_pad

        Arguments
        ---------
        x : torch.Tensor
          Input tensor with original pad_idx
        org_pad : int
          Orginal pad_idx
        custom_pad : int
          Custom pad_idx
        """
        out = x.clone()
        out[x == org_pad] = custom_pad

        return out
