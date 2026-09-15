"""Run a hyperparameter search."""
import sys
from itertools import product
import logging
import jax
from skqd_z2lgt.tasks.open_output import open_output
from skqd_z2lgt.tasks.train_generator import train_single_model

jax.config.update('jax_enable_x64', True)
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


if __name__ == '__main__':
    DATADIR = sys.argv[1]
    JOB = int(sys.argv[2])

    params = open_output(DATADIR)
    params.pkgpath = DATADIR

    num_h_opt = [64, 128, 256, 512]
    l2w_biases_opt = [0.2, 0.6, 1., 1.4]
    batch_size_opt = [32, 64, 96, 128]
    learning_rate_opt = [0.0002, 0.0005, 0.001, 0.002]

    num_h, l2w_biases, batch_size, learning_rate = list(product(
        num_h_opt, l2w_biases_opt, batch_size_opt, learning_rate_opt
    ))[JOB]

    LOG.info('Hyperparameters: num_h=%d, l2w_biases=%.1f, batch_size=%d, lr=%.4f',
             num_h, l2w_biases, batch_size, learning_rate)

    params.crbm.num_h = num_h
    params.crbm.l2w_biases = l2w_biases
    params.crbm.train_batch_size = batch_size
    params.crbm.learning_rate = learning_rate

    train_single_model(params, 0, 3)
