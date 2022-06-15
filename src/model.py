import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics import functional as FM

from rna_gene_dataset import GeneRNADataModule
from attention import TransformerEncoderLayer, TransformerEncoder

from argparse import ArgumentParser

def data_preprocessing(miRNA_vec, Gene_vec, n_split):
    input_vec = torch.cat((miRNA_vec, Gene_vec), axis=-1)
    input_vec = input_vec.view(-1, n_split, input_vec.size(-1) // n_split)
    return input_vec


class MyModel(pl.LightningModule):
    def __init__(self,
                 n_layers,
                 d_model,  # dim. in attemtion mechanism
                 num_heads,  # number of heads
                 learning_rate,
                 n_split
                 ):
        super(MyModel, self).__init__()
        self.save_hyperparameters()

        self.encoder = TransformerEncoder( self.hparams.n_layers,
                                           self.hparams.d_model,
                                            self.hparams.num_heads,
                                            dim_feedforward=self.hparams.d_model*4,
                                            # by convention
                                            dropout=0.1
                                              )


        # [to output]
        self.to_output = nn.Linear(self.hparams.d_model, 1)
        # D -> a single number

        # loss
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, miRNA_vec, Gene_vec):
        # INPUT EMBEDDING
        # [ Digit Character Embedding ]
        # seq_ids : [B, max_seq_len]
        # seq_embs = self.input_emb(seq_ids.long())  # [B, max_seq_len, d_model]

        # ENCODING BY Transformer-Encoder
        # [mask shaping]
        # masking - shape change
        #   mask always applied to the last dimension explicitly.
        #   so, we need to prepare good shape of mask
        #   to prepare [B, dummy_for_heads, dummy_for_query, dim_for_key_dimension]
        # mask = weight[:, None, None, :]  # [B, 1, 1, max_seq_len]
        input_vec = data_preprocessing(miRNA_vec, Gene_vec, self.hparams.n_split)
        seq_encs, attention_scores = self.encoder(input_vec, mask=None)  # [B, max_seq_len, d_model]

        # seq_encs         : [B, max_seq_len, d_model]
        # attention_scores : [B, max_seq_len_query, max_seq_len_key]

        # Output Processing
        # pooling
        blendded_vector = seq_encs[:, 0]  # taking the first(query) - step hidden state

        # To output
        logits = self.to_output(blendded_vector)
        return logits, attention_scores

    def training_step(self, batch, batch_idx):
        miRNA_vec, Gene_vec, label = batch

        logit, _ = self(miRNA_vec, Gene_vec)  # [B, output_vocab_size]
        # loss = self.criterion(logits, y_id.long())
        # preds = logits.argmax(-1)

        loss = self.criterion(logit, label)
        acc = FM.accuracy(logit, label.long(), threshold=0.5)
        auc = FM.auroc(logit, label.long(), pos_label=1)
        log_dict = {"train_loss": loss,
                    "train_acc": acc,
                    "train_auc": auc}
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(log_dict, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        miRNA_vec, Gene_vec, label = batch

        logit, _ = self(miRNA_vec, Gene_vec)  # [B, output_vocab_size]
        # loss = self.criterion(logit, label)

        ## get predicted result
        # prob = F.softmax(logits, dim=-1)
        # acc = FM.accuracy(prob, label.long())
        # auc = FM.auc(prob, label)
        metrics = {'val_logit': logit,
                   'val_label': label}
        # self.log_dict(metrics)
        return metrics

    # def validation_step_end(self, metrics):
    #     print(metrics["val_loss"].mean())

    def validation_epoch_end(self, val_step_outputs):
        val_logit = torch.cat([x['val_logit'] for x in val_step_outputs])
        val_label = torch.cat([x['val_label'] for x in val_step_outputs])

        loss = self.criterion(val_logit, val_label)
        acc = FM.accuracy(val_logit, val_label.long(), threshold=0.5)
        auc = FM.auroc(val_logit, val_label.long(), pos_label=1)

        log_dict = {"val_loss": loss,
                    "val_acc": acc,
                    "val_auc": auc}
        self.log_dict(log_dict, on_epoch=True, prog_bar=True, logger=True)



    def test_step(self, batch, batch_idx):
        miRNA_vec, Gene_vec, label = batch

        logit, _ = self(miRNA_vec, Gene_vec)  # [B, output_vocab_size]
        loss = self.criterion(logit, label)

        ## get predicted result
        # prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(logit, label.long(), threshold=0.5)
        auc = FM.auroc(logit, label.long(), pos_label=1)

        metrics = {'test_acc': acc, 'test_loss': loss, 'test_auc': auc}
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
    parser.add_argument('--d_model', default=32, type=int)  # dim. for attention model
    parser.add_argument('--n_heads', default=4, type=int)  # number of multi-heads
    parser.add_argument('--n_split', default=8, type=int)  # number of data seq
    parser.add_argument('--n_layers', default=8, type=int)  # number of encoder layer

    parser = pl.Trainer.add_argparse_args(parser)
    parser = MyModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = GeneRNADataModule.from_argparse_args(args)
    # iter(dm.train_dataloader()).next()  # <for testing

    # ------------
    # model
    # ------------
    model = MyModel(
        # dm.input_vocab_size,
        # dm.output_vocab_size,
        args.n_layers,
        args.d_model,  # dim. in attemtion mechanism
        args.n_heads,
        # dm.padding_idx,
        args.learning_rate,
        args.n_split
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        max_epochs=100,
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