import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch


def visualize_bayesian_weights(bayesian_layer,
                               num_samples=1000,
                               num_neurons_to_plot=20):
    weight_mu = bayesian_layer.mean.detach().cpu().numpy()
    weight_rho = bayesian_layer.rho.detach().cpu().numpy()
    # breakpoint()
    variance = np.exp(weight_rho * 2)

    # Generating samples
    samples = np.random.normal(
        weight_mu, np.sqrt(variance), size=(num_samples, *weight_mu.shape))

    # Creating a high-resolution plot
    plt.figure(figsize=(15, 7), dpi=300)

    # Plotting distributions for selected neurons
    for i in range(min(num_neurons_to_plot, weight_mu.shape[0])):
        ax = plt.subplot(4, 5, i + 1)
        sns.histplot(
            samples[:, i, :].flatten(),
            bins=30,
            kde=True,
            color='skyblue',
            ax=ax)
        ax.set_title(f'Neuron {i+1}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        # Overlaying a standard normal distribution for comparison
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, np.mean(samples[:, i, :]),
                           np.std(samples[:, i, :]))
        ax.plot(x, p, 'k', linewidth=2)

    plt.tight_layout()
    plt.savefig('bayesian_weights_distribution.png')  # or .png, .svg
    # plt.show()


# Example usage
from piconas.predictor.pinat.model_factory import create_best_nb201_model

model = create_best_nb201_model()
# load model
ckpt_dir = 'checkpoints/nasbench_201/201_cifar10_ParZCBMM_mse_t781_vall_e153_bs10_best_nb201_run2_tau0.783145_ckpt.pt'
model.load_state_dict(torch.load(ckpt_dir, map_location=torch.device('cpu')))

visualize_bayesian_weights(model.encoder.bayesian_mlp_mixer.patch_to_embedding)
