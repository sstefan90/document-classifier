import argparse
import torch
import torch.nn as nn
import copy
import os
import tqdm
from transformers import BertForSequenceClassification
from torch.utils import tensorboard
from utils import TRAIN_FILE_NAME, VAL_FILE_NAME, NUM_LABELS
from utils import process_data
import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"we're using the {DEVICE}!")
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 1
SAVE_INTERVAL = 100
NUM_EPOCHS = 5


def checkpoint_model(model, file_name):
    model.save_pretrained(file_name, from_pt=True)

def fine_tune_bert(model, log_path, writer, batch_size=16, lr=2e-5, max_length=128, mode='normal', patience_limit=1):
    patience = 0
    model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    model.to(DEVICE)

    training_generator = process_data(TRAIN_FILE_NAME, batch_size, max_length)
    validation_generator = process_data(VAL_FILE_NAME, batch_size, max_length)

    # Loop over epochs
    pre_validation_loss = None
    pbar = tqdm.tqdm(range(NUM_EPOCHS))
    for epoch in pbar:
        # Training
        step = 0
        training_loss_list = []
        for X_train, y_train in training_generator:
            # transfer to GPU
            input_ids = X_train['input_ids'].reshape(
                (X_train['input_ids'].shape[0], X_train['input_ids'].shape[-1])).to(DEVICE)

            attention_mask = X_train['attention_mask'].reshape(
                (X_train['attention_mask'].shape[0], X_train['attention_mask'].shape[-1])).to(DEVICE)
            labels = y_train[0].to(DEVICE)

            step += 1
            loss, logits = model(input_ids=input_ids, token_type_ids=None,
                                 attention_mask=attention_mask, labels=labels, return_dict=False)
            training_loss_list.append(loss.detach().cpu())
            if step % PRINT_INTERVAL == 0:
                print(
                    f'epoch: { epoch}, loss: {loss.item()}, logits: {torch.argmax(logits, dim=1)}, y: {y_train}')
                writer.add_scalar('loss/train', loss.item(), epoch*3729 + step)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % VAL_INTERVAL == 0 and step > 0:
                print(f"running validation at step {step}")
                # calculate training loss
                training_loss = sum(training_loss_list) / \
                    len(training_loss_list)
                print(f"current average training loss is {training_loss}")

                training_loss_list = []  # reset training_loss_list
                with torch.inference_mode():
                    validation_loss_list = []
                    val_steps = 0
                    for X_val, y_val in validation_generator:
                        val_steps += 1

                        input_ids = X_val['input_ids'].reshape(
                            (X_val['input_ids'].shape[0], X_val['input_ids'].shape[-1])).to(DEVICE)
                        attention_mask = X_val['attention_mask'].reshape(
                            (X_val['attention_mask'].shape[0], X_val['attention_mask'].shape[-1])).to(DEVICE)
                        labels = y_val[0].to(DEVICE)

                        val_loss, _ = model(input_ids=input_ids, token_type_ids=None,
                                            attention_mask=attention_mask, labels=labels, return_dict=False)
                        validation_loss_list.append(val_loss.detach().cpu())

                    validation_loss = sum(
                        validation_loss_list) / len(validation_loss_list)
                    print(f"validation loss is {validation_loss:.04f}")
                    writer.add_scalar(
                        'loss/val', loss.item(), epoch*3729 + step)

                    # checkpoint the model!
                    checkpoint_file = f"{log_path}/checkpoint/epoch:{epoch}.step:{step}.loss:{validation_loss:.04f}"
                    checkpoint_model(model, checkpoint_file)

                    if pre_validation_loss and (pre_validation_loss - validation_loss) < 0.02 and epoch > 0:
                        # checkpoint the model and break early
                        if patience < patience_limit:
                            patience += 1
                        else:
                            print(
                                f"EARLY STOPPING: validation loss {validation_loss:.04f}, {pre_validation_loss:.04f}")
                            checkpoint_file = f"{log_path}/checkpoint/epoch:{epoch}.step:{step}.loss:{validation_loss:.04f}"
                            checkpoint_model(model, checkpoint_file)
                        return
                    pre_validation_loss = validation_loss
    checkpoint_file = f"{log_path}/checkpoint/epoch:{epoch}.step:{step}.loss:{validation_loss:.04f}"

    checkpoint_model(model, optimizer, checkpoint_file)


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/bert.batch_size:{args.batch_size}.learning_rate:{args.learning_rate}.max_length:{args.max_length}.mode:{args.mode}'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        os.mkdir(log_dir + "/checkpoint")

    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=NUM_LABELS, output_attentions=False, output_hidden_states=False)
    fine_tune_bert(model, log_path=log_dir, writer=writer, batch_size=args.batch_size,
                   lr=args.learning_rate, max_length=args.max_length, mode='normal')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help="optional, define device. If cuda is available, uses GPU resources")
    parser.add_argument('--mode', type=str, default='normal',
                        help="model training mode")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help="learning rate parameter")
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--max_length', type=int, default=128,
                        help='max length of training sample. Script will truncate and pad to max_length param')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=2,
                        help='patience for early stopping')
    args = parser.parse_args()
    main(args)
