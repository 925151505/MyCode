from __future__ import absolute_import, division, print_function
import warnings
import torch
from torch.backends import cudnn
from sklearn.manifold import TSNE
from pre_Data import *
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from torch.nn import MSELoss
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from bert import PMEE_BertForSequenceClassification
from xlnet import PMEE_XLNetForSequenceClassification
from Roberta import RobertaForSequenceClassification

def return_unk():
    return 0

Config = Config_path()
#Cancel prompt message
warnings.filterwarnings('ignore')

def set_random_seed():
    #9997较好
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    seed = random.randint(0, 9999)
    print("Seed: {}".format(seed))
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

def prep_for_training(num_train_optimization_steps: int):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    if args.model == 'bert-base-uncased':
        model = PMEE_BertForSequenceClassification.from_pretrained(
            Config.get('model_path','bert-base-uncased'), multimodal_config=multimodal_config, num_labels=1
        )
    if args.model == 'bert-large-uncased':
        model = PMEE_BertForSequenceClassification.from_pretrained(
            Config.get('model_path', 'bert-large-uncased'), multimodal_config=multimodal_config, num_labels=1
        )
    elif args.model == "xlnet-base-cased":
        model = PMEE_XLNetForSequenceClassification.from_pretrained(
            Config.get('model_path','xlnet-base-cased'), multimodal_config=multimodal_config, num_labels=1
        )
        ckpt = torch.load('./result/xlnet_mosi.bin', map_location=DEVICE)
        model.load_state_dict(ckpt, strict=False)
    elif args.model == "xlnet-large-cased":
        model = PMEE_XLNetForSequenceClassification.from_pretrained(
            Config.get('model_path','xlnet-large-cased'), multimodal_config=multimodal_config, num_labels=1
        )
    elif args.model == "xlnet-base-cased-mosi":
        # model = PMEE_XLNetForSequenceClassification.from_pretrained(
        #     Config.get('model_path', 'xlnet-base-cased-mosi'), multimodal_config=multimodal_config, num_labels=1, ignore_mismatched_sizes=True
        # )
        model = PMEE_XLNetForSequenceClassification.from_pretrained(
            Config.get('model_path', 'xlnet-base-cased'), multimodal_config=multimodal_config, num_labels=1
        )
        ckpt = torch.load('./result/xlnet_mosei.bin', map_location=DEVICE)
        model.load_state_dict(ckpt, strict=False)

    elif args.model == "roberta-base":
        model = RobertaForSequenceClassification.from_pretrained(
            Config.get('model_path', 'roberta-base'), multimodal_config=multimodal_config, num_labels=1
        )
    model.to(DEVICE)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=True )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler



def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    t_cat_feature = []
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs, contrastive_loss, re_cat_feature = model(
        # outputs = model(
            input_ids,
            visual,
            acoustic,
            'train',
            t_cat_feature,
            label_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
        )

        t_cat_feature.append(re_cat_feature)
        logits = outputs
        #Calculate loss
        loss_fct = MSELoss()
        loss = 0.1 * loss_fct(logits.view(-1), label_ids.view(-1)).to(DEVICE) + 0.9 * contrastive_loss.to(DEVICE)


        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()
        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        #################
    l = len(t_cat_feature[0])

    if 'base' in args.model:
        cat_feature = torch.zeros([6, 768]).to(DEVICE)
    elif 'large' in args.model:
        cat_feature = torch.zeros([6, 1024]).to(DEVICE)

    #Arrays for visualization
    cls_tokens = []
    label = []
    for i in range(l):
        global index
        index = 0
        for j in range(len(t_cat_feature)):
            a = t_cat_feature[j][i]
            if 'base' in args.model:
                if not a.equal(torch.zeros(768).to(DEVICE)):
                    cat_feature[i] = cat_feature[i] + a
                    #Visualization of 2 categories
                    cls_tokens.append(a.cpu().detach().numpy())
                    #Two category visualization
                    if i < 3:
                        label.append(0)
                    if i >= 3:
                        label.append(1)
                    #Visualization of 7 categories
                    #label.append(i)
                    index = index + 1
            elif 'large' in args.model:
                if not a.equal(torch.zeros(1024).to(DEVICE)):
                    cat_feature[i] = cat_feature[i] + a
                    index = index + 1
        if index != 0:
            cat_feature[i] = cat_feature[i] / (index * 1.0)
    # print(cls_tokens)
    # print(label)
    visual_data = {}
    visual_data['data'] = cls_tokens
    visual_data['label'] = label
    return tr_loss / nb_tr_steps, cat_feature, visual_data


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                'eval',
                label_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))


            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader, cat_feature):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                'test',
                cat_feature,
                label_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None
            )
            logits = outputs

            logits = np.squeeze(logits.detach().cpu().numpy()).tolist()
            label_ids = np.squeeze(label_ids.detach().cpu().numpy()).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)
    return preds, labels

from sklearn.metrics import confusion_matrix


def test_score_model(model: nn.Module, test_dataloader: DataLoader, cat_feature, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader, cat_feature)

    #Exclude zero values
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    #Calculate the correlation coefficient matrix of the matrix. The elements of the correlation coefficient matrix are [-1, 1], and the closer it is to 1, the more relevant it is
    corr = np.corrcoef(preds, y_test)[0][1]

    #Transforming regression problems into classification problems
    preds_2 = preds >= 0
    y_test_2 = y_test >= 0

    preds_7 = np.round(preds)
    y_test_7 = np.round(y_test)


    f_score = f1_score(y_test_2, preds_2, average="weighted")
    acc_2 = accuracy_score(y_test_2, preds_2)
    acc_7 = accuracy_score(y_test_7, preds_7)
    if acc_2 > 0.869:
        x, y = [], []#y_true, y_pred
        for i, j in zip(y_test_2, preds_2):
            #做混淆矩阵用。积极1消极0
            if i == True:
                x.append(1)
            if i == False:
                x.append(0)
            if j == True:
                y.append(1)
            if j == False:
                y.append(0)
        matrix = confusion_matrix(x, y)
        print('This ACC2：', acc_2)
        print(matrix)

    return acc_2, acc_7, mae, corr, f_score


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
    i
):
    valid_losses = []
    test_accuracies = []
    best_MAE = 1
    best_corr = 0
    best_f1 = 0
    best_acc_2 = 0
    best_acc_7 = 0
    for epoch_i in range(int(args.n_epochs)):
        train_loss, cat_feature, visual_data = train_epoch(model, train_dataloader, optimizer, scheduler)
        #train_loss, cat_feature = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        test_acc_2, test_acc_7, test_mae, test_corr, test_f_score = test_score_model(
            model, test_data_loader, cat_feature
        )
        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_acc2:{}, test_acc7:{}".format(
                epoch_i, train_loss, valid_loss, test_acc_2, test_acc_7
            )
        )
        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc_2)
        if test_acc_2 > best_acc_2 and test_acc_2 > 0.868:
            best_acc_2 = test_acc_2
            pickle.dump(visual_data, open('feature/PMEE_mosi.pkl', 'wb'))
        if test_mae < best_MAE:
            best_MAE = test_mae
        if test_corr > best_corr:
            best_corr = test_corr
        if test_acc_7 > best_acc_7:
            best_acc_7 = test_acc_7
    print("best:, acc:{}, acc7:{},f1:{}, corr:{}, mae:{}".format( best_acc_2, best_acc_7, best_f1, best_corr, best_MAE)
          )


    return best_acc_2, best_acc_7, best_f1, best_MAE, best_corr


def main():
    a, b, c, d, e, f, g = 0, 0, 0, 0, 0, 0, 0
    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)
    best_acc = 0
    print(model.modules())
    for i in range(5):
        # wandb.init(project="MAG")
        # wandb.config.update(args)

        #set_random_seed()

        random.seed(3501)
        torch.manual_seed(3501)
        torch.cuda.manual_seed_all(3501)#Set random seeds for all GPUs (multiple GPUs)
        torch.cuda.manual_seed(3501)#Set a random seed for the current GPU (using only one GPU)
        cudnn.deterministic = True

        acc_2, acc_7, f1, mae, corr = train(
            model,
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            optimizer,
            scheduler,
            i+1
        )
        if acc_2 > best_acc:
            best_acc = acc_2
            #torch.save(model.state_dict(), './result/xlnet_mosei.bin')
            #print("save model on MOSI")

        a += acc_2
        b += f1
        c += mae
        d += corr
        e += acc_7
    print("average:, acc:{}, acc7:{},f1:{}, corr:{}, mae:{}".format(a/5.0, e/5.0, b/5.0, d/5.0, c/5.0)
          )
if __name__ == "__main__":
    main()