from lib.evaluator import evaluate
from lib.reinforce import PolicyGradientTrainer


def main():
    device = "cpu"
    net = PolicyGradientTrainer.train_new_net()
    net.print_summary(device)
    evaluate("policy gradient", net, 10, device)


if __name__ == "__main__":
    main()
