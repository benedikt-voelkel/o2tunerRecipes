"""
Skeleton for evaluation
"""
import matplotlib.pyplot as plt

from o2tuner.log import get_logger


def evaluate(inspectors, config):

    valueN = config["parameterN"]
    get_logger().info("Evaluation...\nParameter1=%s", valueN)

    for i, insp in enumerate(inspectors):
        figure, _ = insp.plot_loss_feature_history()
        figure.tight_layout()
        figure.savefig(f"loss_feature_history_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_importance()
        figure.tight_layout()
        figure.savefig(f"importance_parameters_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_parallel_coordinates()
        figure.savefig(f"parallel_coordinates_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_slices()
        figure.savefig(f"slices_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_correlations()
        figure.savefig(f"parameter_correlations_{i}.png")
        plt.close(figure)

        figure, _ = insp.plot_pairwise_scatter()
        figure.savefig(f"pairwise_scatter_{i}.png")
        plt.close(figure)

    # return True if everything went well. Can also return False at any other point to indicate some kind of failure.
    return True
