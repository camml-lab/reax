Data Handling
=============

REAX provides flexible tools for managing data using :class:`~reax.data.DataLoader` and
:class:`~reax.DataModule`.

DataLoaders
-----------

REAX works seamlessly with JAX-compatible data loaders. You can use standard PyTorch DataLoaders
or any iterable that yields batches of numpy/jax arrays.

DataModules
-----------

A :class:`~reax.DataModule` encapsulates all steps needed to process data: downloading,
tokenising, and splitting. It ensures reproducibility and makes data handling reusable across
projects.

A DataModule is defined by 5 steps:

1.  **prepare_data**: Download, tokenise, etc. (runs only on 1 CPU in distributed settings).
2.  **setup**: Split data, apply transforms (runs on every device).
3.  **train_dataloader**: Returns the training dataloader.
4.  **val_dataloader**: Returns the validation dataloader.
5.  **test_dataloader**: Returns the test dataloader.

Example
-------

.. code-block:: python

    class MNISTDataModule(reax.DataModule):
        def prepare_data(self):
            # Download MNIST
            pass

        def setup(self, stage=None):
            # Split dataset
            pass

        def train_dataloader(self):
            return DataLoader(...)

        def val_dataloader(self):
            return DataLoader(...)

Using a DataModule
------------------

Pass the DataModule to the Trainer:

.. code-block:: python

    dm = MNISTDataModule()
    trainer.fit(model, datamodule=dm)
