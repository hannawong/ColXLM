from colXLM.utils.parser import Arguments
from colXLM.utils.runs import Run
from colXLM.training.training import train


def main():
    parser = Arguments(description='Training with <query, positive passage, negative passage>.')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()

    args = parser.parse()

    with Run.context(consider_failed_if_interrupted=False):
        train(args)


if __name__ == "__main__":
    main()
