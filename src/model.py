import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics import functional as FM

from gene_default_dataset import GeneRNADataModule
from attention import TransformerEncoderLayer


gn_data_module = GeneRNADataModule(batch_size=32, train_ratio=0.8)
miRNA_vec, Gene_vec, label = iter(gn_data_module.train_dataloader()).__next__()

print(gn_data_module)

d_model = 256
num_heads=8
encoder = TransformerEncoderLayer( d_model,
                                    num_heads,
                                    dim_feedforward=d_model*4,
                                    # by convention
                                    dropout=0.1)
inputs = torch.cat((miRNA_vec, Gene_vec), axis=-1)

wq = nn.Linear(d_model, d_model, bias=use_bias)

encoder(x=inputs, mask=None)


class MyModel(pl.LightningModule):
    def __init__(self,
                 d_model,  # dim. in attemtion mechanism
                 num_heads,  # number of heads
                 ):
        super.__init__()
        self.save_hyperparameters()

        self.encoder = TransformerEncoderLayer( self.hparams.d_model,
                                                self.hparams.num_heads,
                                                dim_feedforward=self.hparams.d_model*4,
                                                # by convention
                                                dropout=0.1
                                              )


        # [to output]
        self.to_output = nn.Linear(self.hparams.d_model, self.hparams.output_vocab_size)
        # D -> a single number

        # loss
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, seq_ids, weight):
        # INPUT EMBEDDING
        # [ Digit Character Embedding ]
        # seq_ids : [B, max_seq_len]
        seq_embs = self.input_emb(seq_ids.long())  # [B, max_seq_len, d_model]

        # ENCODING BY Transformer-Encoder
        # [mask shaping]
        # masking - shape change
        #   mask always applied to the last dimension explicitly.
        #   so, we need to prepare good shape of mask
        #   to prepare [B, dummy_for_heads, dummy_for_query, dim_for_key_dimension]
        mask = weight[:, None, None, :]  # [B, 1, 1, max_seq_len]
        seq_encs, attention_scores = self.encoder(seq_embs, mask)  # [B, max_seq_len, d_model]

        # seq_encs         : [B, max_seq_len, d_model]
        # attention_scores : [B, max_seq_len_query, max_seq_len_key]

        # Output Processing
        # pooling
        blendded_vector = seq_encs[:, 0]  # taking the first(query) - step hidden state

        # To output
        logits = self.to_output(blendded_vector)
        return logits, attention_scores

    def training_step(self, batch, batch_idx):
        seq_ids, weights, y_id = batch
        logits, _ = self(seq_ids, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        seq_ids, weights, y_id = batch

        logits, _ = self(seq_ids, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long())

        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_acc = val_step_outputs['val_acc'].cpu()
        val_loss = val_step_outputs['val_loss'].cpu()

        self.log('validation_acc', val_acc, prog_bar=True)
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq_ids, weights, y_id = batch

        logits, _ = self(seq_ids, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long())

        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics, on_epoch=True)
        return metrics

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ATTENTION")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--d_model', default=512, type=int)  # dim. for attention model
    parser.add_argument('--num_heads', default=8, type=int)  # number of multi-heads

    parser = pl.Trainer.add_argparse_args(parser)
    parser = TransformerEncoder_Number_Finder.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = NumberDataModule.from_argparse_args(args)
    iter(dm.train_dataloader()).next()  # <for testing

    # ------------
    # model
    # ------------
    model = MyModel(
        # dm.input_vocab_size,
        # dm.output_vocab_size,
        args.d_model,  # dim. in attemtion mechanism
        args.num_heads,
        dm.padding_idx,
        args.learning_rate
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        max_epochs=2,
        # callbacks=[EarlyStopping(monitor='val_loss')],
        gpus=1  # if you have gpu -- set number, otherwise zero
    )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, datamodule=dm)
    print(result)



if __name__ == '__main__':
    cli_main()