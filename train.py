import os
from dataset import Sacarsm
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm, trange
from sklearn import metrics
import wandb
import numpy as np

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from model import MV_CLIP
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
import wandb
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_data, dev_data, test_data, processor):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              collate_fn=Sacarsm.collate_func,
                              shuffle=True)
    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device)

    if args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule

        print('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            model.parameters(),
            # lr=1e-3,
            # eps=(1e-30, 1e-3),
            # clip_threshold=1.0,
            # decay_rate=-0.8,
            # beta1=None,
            lr=None,
            weight_decay=args.weight_decay,
            relative_step=True,
            scale_parameter=True,
            warmup_init=True
        )
        scheduler = AdafactorSchedule(optimizer)
    elif args.optimizer_name == 'adam':
        print('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        if args.model == 'MV_CLIP':
            clip_params = list(map(id, model.model.parameters()))
            base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
            optimizer = AdamW([
                    {"params": base_params},
                    {"params": model.model.parameters(),"lr": args.clip_learning_rate}
                    ], lr=args.learning_rate, weight_decay=args.weight_decay)

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                    num_training_steps=total_steps)
        else:
            optimizer = optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    else:
        raise Exception('Wrong Optimizer Name!!!')


    max_acc = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0.
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()

        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list = batch
            if args.model == 'MV_CLIP':
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device)

            loss, score = model(inputs,labels=labels)
            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            if args.optimizer_name == 'adam':
                scheduler.step() 
            optimizer.zero_grad()

        wandb.log({'train_loss': sum_loss/sum_step})
        dev_acc, dev_f1 ,dev_precision,dev_recall = evaluate_acc_f1(args, model, device, dev_data, processor, mode='dev')
        wandb.log({'dev_acc': dev_acc, 'dev_f1': dev_f1, 'dev_precision': dev_precision, 'dev_recall': dev_recall})
        logging.info("i_epoch is {}, dev_acc is {}, dev_f1 is {}, dev_precision is {}, dev_recall is {}".format(i_epoch, dev_acc, dev_f1, dev_precision, dev_recall))

        if dev_acc > max_acc:
            max_acc = dev_acc

            path_to_save = os.path.join(args.output_dir, args.model)
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, 'model.pt'))

            test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(args, model, device, test_data, processor,macro = True, mode='test')
            _, test_f1_,test_precision_,test_recall_ = evaluate_acc_f1(args, model, device, test_data, processor, mode='test')
            wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1,
                     'macro_test_precision': test_precision,'macro_test_recall': test_recall, 'micro_test_f1': test_f1_,
                     'micro_test_precision': test_precision_,'micro_test_recall': test_recall_})
            logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(i_epoch, test_acc, test_f1, test_precision, test_recall, test_f1_, test_precision_, test_recall_))

        torch.cuda.empty_cache()
    logger.info('Train done')


def evaluate_acc_f1(args, model, device, data, processor, macro=False,pre = None, mode='test'):
        data_loader = DataLoader(data, batch_size=args.dev_batch_size, collate_fn=Sacarsm.collate_func,shuffle=False)
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        model.eval()
        sum_loss = 0
        sum_step = 0
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                text_list, image_list, label_list, id_list = t_batch
                if args.model == 'MV_CLIP':
                    inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                    labels = torch.tensor(label_list).to(device)
                
                t_targets = labels
                loss, t_outputs = model(inputs,labels=labels)
                sum_loss += loss.item()
                sum_step += 1
  
                outputs = torch.argmax(t_outputs, -1)

                n_correct += (outputs == t_targets).sum().item()
                n_total += len(outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)
        if mode == 'test':
            wandb.log({'test_loss': sum_loss/sum_step})
        else:
            wandb.log({'dev_loss': sum_loss/sum_step})
        if pre != None:
            with open(pre,'w',encoding='utf-8') as fout:
                predict = t_outputs_all.cpu().numpy().tolist()
                label = t_targets_all.cpu().numpy().tolist()
                for x,y,z in zip(predict,label):
                    fout.write(str(x) + str(y) +z+ '\n')
        if not macro:   
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu())
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu())
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu())
        else:
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1],average='macro')
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
        return acc, f1 ,precision,recall

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device number')
    parser.add_argument('--model', default='MV_CLIP', type=str, help='the model name', choices=['MV_CLIP'])
    parser.add_argument('--text_name', default='text_data', type=str, help='the text data folder name')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='number of train epoched')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for modules expect CLIP')
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP')
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--output_dir', default='../output_dir/', type=str, help='the output path')
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    seed_everything(args.seed)

    wandb.init(
        project="MMSD2.0",
        notes="mm",
        tags=["mm"],
        config=vars(args),
    )
    wandb.watch_called = False  

    train_data = Sacarsm(mode='train', text_name=args.text_name, limit=None)
    dev_data = Sacarsm(mode='valid', text_name=args.text_name, limit=None)
    test_data = Sacarsm(mode='test', text_name=args.text_name, limit=None)
    print(len(train_data))

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = MV_CLIP(args)

    model.to(device)
    wandb.watch(model, log="all")

    train(args, model, device, train_data, dev_data, test_data, processor)

if __name__ == '__main__':
    main()
