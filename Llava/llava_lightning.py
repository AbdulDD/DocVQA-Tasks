# Imports
import pytorch_lightning as pl

class LlavaTraining(pl.LightningModule):
    def __init__(self, config, model, processor):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.model.train()

    def training_step(self, batch, batch_idx):

        # 1. Extract the inputs
        input_ids = batch["input_ids"]         # [B, 1024]
        attention_mask = batch["attention_mask"]  # [B, 1024]
        pixel_values = batch["pixel_values"]      # [B, 3, 336, 336]
        labels = batch["labels"]                 # [B, 1024]

        # 2. Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        # 3. Log the loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"]   # FIXED
        )
        return optimizer
