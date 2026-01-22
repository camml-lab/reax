REAX
====

.. image:: https://codecov.io/gh/muhrin/reax/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/muhrin/reax
    :alt: Coverage

.. image:: https://github.com/camml-lab/reax/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/camml-lab/reax/actions/workflows/ci.yml
    :alt: Tests

.. image:: https://readthedocs.org/projects/reax/badge/
    :target: https://reax.readthedocs.io/
    :alt: Documentation

.. image:: https://img.shields.io/pypi/v/reax.svg
    :target: https://pypi.python.org/pypi/reax/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/reax.svg
    :target: https://pypi.python.org/pypi/reax/

.. image:: https://img.shields.io/pypi/l/reax.svg
    :target: https://pypi.python.org/pypi/reax/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black


REAX — Scalable, flexible training for JAX, inspired by the simplicity of PyTorch Lightning.

REAX - Scalable Training for JAX
================================

REAX is a minimal and high-performance framework for training JAX models, designed to simplify
research workflows. Inspired by PyTorch Lightning, it brings similar high-level abstractions and
scalability to JAX users, making it easier to scale models across multiple GPUs with minimal
boilerplate. 🚀

A Port of PyTorch Lightning to JAX
----------------------------------

Much of REAX is built by porting the best practices and abstractions of **PyTorch Lightning** to
the **JAX** ecosystem. If you're familiar with PyTorch Lightning, you'll recognize concepts like:

- Training loops ⚡
- Multi-GPU training 🖥️
- Logging and checkpointing 💾

However, REAX has been designed with JAX-specific optimizations, ensuring high performance without
sacrificing flexibility.

Why REAX? 🌟
------------

- **Scalable**: Built to leverage JAX’s parallelism and scalability. ⚡
- **Minimal Boilerplate**: Simplifies the training process with just enough structure. 🧩
- **Familiar**: For users who have experience with frameworks like PyTorch Lightning, the
  transition to REAX is seamless. 🔄

Installation 🛠️
---------------

To install REAX, run the following command:

.. code-block:: shell

    pip install reax


REAX example
------------

Define the training workflow. Here's a toy example:

.. code-block:: python

    # main.py
    from functools import partial
    import jax, optax, reax, flax.linen as linen
    from reax.demos import mnist


    class Autoencoder(linen.Module):
        def setup(self):
            super().__init__()
            self.encoder = linen.Sequential([linen.Dense(128), linen.relu, linen.Dense(3)])
            self.decoder = linen.Sequential([linen.Dense(128), linen.relu, linen.Dense(28 * 28)])

        def __call__(self, x):
            z = self.encoder(x)
            return self.decoder(z)


    # --------------------------------
    # Step 1: Define a REAX Module
    # --------------------------------
    # A ReaxModule (nn.Module subclass) defines a full *system*
    # (ie: an LLM, diffusion model, autoencoder, or simple image classifier).
    class ReaxAutoEncoder(reax.Module):
        def __init__(self):
            super().__init__()
            self.ae = Autoencoder()

        def configure_model(self, stage: reax.Stage, batch, /):
            if self.parameters() is None:
                # Prepare example batch for initialization
                x, _ = batch
                x = x.reshape(x.shape[0], -1)
                # Initialize Flax Linen model with RNGs and example input
                params = self.ae.init(self.rngs(), x)
                self.set_parameters(params)

        def __call__(self, x):
            # Use the full autoencoder model for forward pass
            return self.ae.apply(self.parameters(), x)

        def training_step(self, batch, batch_idx):
            x, _ = batch
            x = x.reshape(x.shape[0], -1)
            # Static method receives params, data, and model apply function
            loss, grads = jax.value_and_grad(self.loss_fn, argnums=0)(
                self.parameters(), x, self.ae.apply
            )
            self.log("train_loss", loss, on_step=True, prog_bar=True)
            return loss, grads

        @staticmethod
        @partial(jax.jit, static_argnums=2)
        def loss_fn(params, x, apply_fn):
            """Compute reconstruction loss.

            Static method for JIT compilation. Receives parameters and apply function.
            """
            predictions = apply_fn(params, x)
            return optax.losses.squared_error(predictions, x).mean()

        def configure_optimizers(self):
            opt = optax.adam(learning_rate=1e-3)
            state = opt.init(self.parameters())
            return opt, state


    # -------------------
    # Step 2: Define data
    # -------------------
    dataset = mnist.MnistDataset(download=True)
    trainer = reax.Trainer()
    train, val = reax.data.random_split(trainer.rngs, dataset, [55000, 5000])

    # -------------------
    # Step 3: Train
    # -------------------
    autoencoder = ReaxAutoEncoder()
    trainer.fit(autoencoder, reax.ReaxDataLoader(train), reax.ReaxDataLoader(val))

Here, we reproduce an example from PyTorch Lightning, so we use torch vision to fetch the data,
but for real models there's no need to use this or pytorch at all.


Disclaimer ⚠️
-------------

REAX takes inspiration from PyTorch Lightning, and large portions of its core functionality are
directly ported from Lightning. If you are already familiar with Lightning, you'll feel right at
home with REAX, but we’ve tailored it to work seamlessly with JAX's performance optimizations.
