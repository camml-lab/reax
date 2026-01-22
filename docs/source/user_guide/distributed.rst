Distributed Training
====================

REAX makes it easy to scale your training to multiple GPUs or TPUs.

Strategies
----------

REAX supports several distributed strategies:

*   **'ddp'** (Data Distributed Parallel): Replicates the model on each device and synchronises
    gradients.
*   **'fsdp'** (Fully Sharded Data Parallel): Shards the model parameters across devices to save
    memory.
*   **'auto'**: Automatically selects the best strategy based on the available hardware.

Configuration
-------------

To enable distributed training, simply set the ``devices`` and ``strategy`` arguments in the
Trainer:

.. code-block:: python

    # Train on 4 GPUs using DDP
    trainer = reax.Trainer(accelerator="gpu", devices=4, strategy="ddp")

Launch Methods
--------------

You can launch your script using standard tools like ``mpirun`` or SLURM. REAX will automatically
detect the environment and initialise the distributed backend.
