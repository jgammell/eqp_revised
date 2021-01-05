from framework import eqp
from framework import datasets
from matplotlib import pyplot as plt
import os

topology = \
{
    'layer sizes': [6, 5, 4, 4, 3],
    'network type': 'SW_intra',#'MLFF',
    'bypass p': .1,
    'bypass mag': .05
}
hyperparameters = \
{
    'learning rate': .02,
    'epsilon': .03,
    'beta': .04,
    'free iterations': 100,
    'weakly clamped iterations': 20
}
configuration = \
{
    'batch size': 20,
    'training examples': 100,
    'test examples': 100,
    'device': 'cpu',
    'seed': 1
}

Network = eqp.Network(topology, hyperparameters, configuration, datasets.MNIST)
(fig, ax) = plt.subplots(1, 2)
ax[0].imshow(Network.W.squeeze().numpy(), vmin=-.5, vmax=.5)
ax[1].imshow(Network.W_mask.squeeze().numpy(), vmin=0, vmax=1)
fig.savefig(os.path.join(os.getcwd(), '..', 'results', 'validate_code_figures', 'weight_matrices.png'))
(fig, ax) = plt.subplots(1, len(Network.interlayer_connections))
for i, conn in zip(range(len(Network.interlayer_connections)), Network.interlayer_connections):
    ax[i].imshow(conn.squeeze().numpy(), vmin=0, vmax=1)
fig.savefig(os.path.join(os.getcwd(), '..', 'results', 'validate_code_figures', 'connection_masks.png'))

