"""
Run Memotion 3_class_memes on pretrained ViLT
"""
import os
from transformers import ViltProcessor
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score

from datasets.ViLT_memotion_dataset import load_dataset
from models.ViLT import ViLTForMemeSentimentClassification

vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = vilt_processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)

    return batch


def setup_dataloader(args, train_dataset, val_dataset):
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, val_dataloader


def setup_model(args):
    if args.checkpoint:
        model = torch.load(os.path.join(args.output_dir, args.checkpoint))
    else:
        model = ViLTForMemeSentimentClassification(args.n_sentiment)
    return model


def setup_optimizer(args, model, device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    return criterion, optimizer


def train_epoch(model, train_dataloader, device, optimizer, criterion, is_train=True):
    if is_train:
        model.train()
    epoch_loss = 0
    sentiment_preds = []
    sentiment_labels = []
    for batch in tqdm(train_dataloader):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].squeeze().to(device)

        # forward + backward + optimize
        pred_logits = model(inputs)
        loss = criterion(pred_logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        sentiment_pred = pred_logits.argmax(-1)
        sentiment_preds.extend(sentiment_pred.cpu())
        sentiment_labels.extend(labels.cpu())

    sentiment_acc = accuracy_score(sentiment_preds, sentiment_labels)
    epoch_loss /= len(sentiment_labels)
    return sentiment_acc, epoch_loss


def validate(model, val_dataloader, device, optimizer, criterion):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            model, val_dataloader, device, optimizer, criterion, is_train=False
        )

    return val_loss, val_acc


def train_model(model, train_dataloader, val_dataloader, device, n_epoch, optimizer, criterion):
    for epoch in range(n_epoch):  # loop over the 3_class_memes
        print(f"Epoch: {epoch}")
        sentiment_acc, epoch_loss = train_epoch(model, train_dataloader, device, optimizer, criterion)
        print(
            f"train acc : {sentiment_acc} | train loss: {epoch_loss}"
        )

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                model,
                val_dataloader,
                device,
                optimizer,
                criterion
            )
            print(f"val acc : {val_acc} | val loss: {val_loss}")

        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "ViLT_model_w_memotion.ckpt")
            file = open(ckpt_file, 'w+')
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)
            file.close()


def main(args):
    # Get Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load 3_class_memes
    train_dataset, val_dataset = load_dataset(args.dataset_filepath)

    # Get Dataloader
    train_dataloader, val_dataloader = setup_dataloader(args, train_dataset, val_dataset)

    # Set up model
    model = setup_model(args)
    model.to(device)

    # Setup loss criterion and optimizer
    criterion, optimizer = setup_optimizer(args, model, device)

    train_model(model, train_dataloader, val_dataloader, device, args.n_epoch, optimizer, criterion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=256, help="size of each batch in loader"
    )
    parser.add_argument(
        "--n_epoch", type=int, default=10, help="total number of epoch"
    )
    parser.add_argument(
        "--n_sentiment", type=int, help="total number of sentiment categories"
    )
    parser.add_argument(
        "--save_every",
        default=2,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="output/ViLT/",
        type=str,
        help="output directory name",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="filename of model checkpoint",
    )
    parser.add_argument(
        "--val_every",
        default=2,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--dataset_filepath",
        type=str,
        help="filepath and name of pickle data file",
    )
    args = parser.parse_args()
    main(args)