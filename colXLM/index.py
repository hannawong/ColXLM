import os
import ujson
import random

from colXLM.utils.runs import Run
from colXLM.utils.parser import Arguments
import colXLM.utils.distributed as distributed

from colXLM.utils.utils import print_message, create_directory
from colXLM.indexing.encoder import CollectionEncoder


def main():
    random.seed(12345)

    parser = Arguments(description='Precomputing document representations with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_indexing_input()

    parser.add_argument('--chunksize', dest='chunksize', default=6.0, required=False, type=float)   # in GiBs

    args = parser.parse()

    with Run.context():
        args.index_path = os.path.join(args.index_root, args.index_name)
        assert not os.path.exists(args.index_path), args.index_path

        distributed.barrier(args.rank)

        if args.rank < 1:
            create_directory(args.index_root)
            create_directory(args.index_path)

        distributed.barrier(args.rank)

        process_idx = max(0, args.rank)
        encoder = CollectionEncoder(args, process_idx=process_idx, num_processes=args.nranks)
        encoder.encode()

        distributed.barrier(args.rank)

        # Save metadata.
        if args.rank < 1:
            metadata_path = os.path.join(args.index_path, 'metadata.json')
            print_message("Saving (the following) metadata to", metadata_path, "..")
            print(args.input_arguments)

            with open(metadata_path, 'w') as output_metadata:
                ujson.dump(args.input_arguments.__dict__, output_metadata)

        distributed.barrier(args.rank)


if __name__ == "__main__":
    main()

# TODO: Add resume functionality
