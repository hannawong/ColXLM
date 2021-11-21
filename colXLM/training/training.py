import os
import random
import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW

from colXLM.utils.amp import MixedPrecisionManager
from colXLM.training.eager_batcher import EagerBatcher
from colXLM.parameters import DEVICE
from colXLM.modeling.colbert import ColBERT
from colXLM.utils.utils import print_message
from colXLM.training.utils import get_mask, print_progress, manage_checkpoints, shuf_order

mode = "BERT"
def train(args):

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)
    
    languages = args.langs.split(",")
    reader_dic = {}
    for lang in languages:
        reader_dic[lang] = EagerBatcher(args, os.path.join(args.triples, f"triples.train.{lang}.tsv"),
                                        (0 if args.rank == -1 else args.rank), args.nranks)
    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    colbert = ColBERT.from_pretrained('bert-base-multilingual-uncased',
                                            query_maxlen=32,
                                            doc_maxlen=180,
                                            dim=128,
                                            similarity_metric="l2",
                                            mask_punctuation=True)


    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)


    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)

    def train_lang(lang, step):
        
        print(f"training with language {lang}")
        train_loss = 0.0
        train_loss_qlm = 0.0

        start_batch_idx = 0
        labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)
        optimizer.zero_grad()
        reader = reader_dic[lang]

        if args.resume:
            assert args.checkpoint is not None
            start_batch_idx = checkpoint['batch']

            reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

        for batch_idx, BatchSteps in zip(range(start_batch_idx, 1), reader):
        
            for queries, passages in BatchSteps:
            ##qlm             
                with amp.context():
                    queries_qlm,passage_qlm,labels_qlm = get_mask(queries,passages,args,reader)
                    loss = colbert(queries_qlm,passage_qlm,"qlm",labels_qlm)

                amp.backward(loss)
                train_loss_qlm += loss.item()

            avg_loss = train_loss_qlm / (batch_idx+1)
            print_message(step, avg_loss)
            amp.step(colbert, optimizer)
            

            ###rr         
            for queries, passages in BatchSteps:
                with amp.context():
                    scores = colbert(queries, passages,"rr").view(2, -1).permute(1, 0)
                    loss = criterion(scores, labels[:scores.size(0)])

                print_progress(scores)
                amp.backward(loss)
                train_loss += loss.item()

            amp.step(colbert, optimizer)
            
            avg_loss = train_loss / (batch_idx+1)
            print_message(step, avg_loss)
        

    for step in range(args.maxsteps):
        for lang in shuf_order(languages):
            train_lang(lang,step)
        manage_checkpoints(args, colbert, optimizer, step+1)
