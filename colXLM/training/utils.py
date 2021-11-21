import os
import torch
import random
from colXLM.utils.runs import Run
from colXLM.utils.utils import save_checkpoint
from colXLM.parameters import SAVED_CHECKPOINTS


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(args, colbert, optimizer, batch_idx):
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)

    if batch_idx % args.save_step == 0:
        name = os.path.join(path, "colbert.dnn")
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

    if batch_idx in SAVED_CHECKPOINTS:
        name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

def get_mask(queries,passages,args,reader):
    queries_qlm = (queries[0][:args.bsize],queries[1][:args.bsize])
    passage_qlm = (passages[0][:args.bsize],passages[1][:args.bsize])

    input_ids = queries_qlm[0]

    probability_matrix = torch.full(input_ids.shape, args.mlm_probability)
    query_mask = (queries_qlm[0] == 101) | (queries_qlm[0] == 100)|(queries_qlm[0] == 103)
    probability_matrix.masked_fill_(query_mask, value=0.0)
            
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels_qlm = input_ids.clone()
    labels_qlm[~masked_indices] = -100 # We only compute loss on masked tokens
    probability_matrix_doc = torch.full(passage_qlm[0].shape, -100)
    labels_qlm = torch.cat([labels_qlm,probability_matrix_doc],axis = 1)

    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = reader.query_tokenizer.mask_token_id
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(reader.query_tokenizer), input_ids.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    queries_qlm = (input_ids,queries_qlm[1])
    
    return queries_qlm,passage_qlm,labels_qlm

def shuf_order(langs):
    """
    Randomize training order.
    """
    tmp = langs
    random.shuffle(tmp)
    return tmp