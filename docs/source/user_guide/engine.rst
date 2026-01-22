The Engine
==========

The :class:`~reax.Engine` is the low-level execution component of REAX. While the
:class:`~reax.Trainer` provides a strict, opinionated structure, the Engine offers flexible
primitives for building custom training loops while still benefiting from REAX's distributed
capabilities.

Trainer vs. Engine
------------------

*   **Trainer**: High-level, "batteries included". Manages loops, logging, checkpointing, and
    listeners automatically. Best for standard training workflows.
*   **Engine**: Low-level, explicit. You write the loops. It handles device placement, distributed
    communication, and optimizer wrapping. Best for research and non-standard loops.

Initialising the Engine
-----------------------

.. code-block:: python

    engine = reax.Engine(accelerator="gpu", devices=4, strategy="ddp")

Using the Engine
----------------

The Engine provides helper methods to setup your environment.

Setup
~~~~~

:meth:`~reax.Engine.setup` prepares your model and optimizers for distributed training.

.. code-block:: python

    model, optimizer = engine.setup(model, optimizer)

Setup DataLoaders
~~~~~~~~~~~~~~~~~

:meth:`~reax.Engine.setup_dataloaders` prepares your dataloaders, handling sharding and device
placement.

.. code-block:: python

    train_loader = engine.setup_dataloaders(train_loader)

Distributed Primitives
~~~~~~~~~~~~~~~~~~~~~~

The Engine exposes methods for cross-device communication:

*   :meth:`~reax.Engine.all_reduce`: Average or sum a tensor across all devices.
*   :meth:`~reax.Engine.broadcast`: Send a tensor from one device to all others.
*   :meth:`~reax.Engine.barrier`: Synchronize all processes.

Example Custom Loop
-------------------

.. code-block:: python

    engine = reax.Engine()
    model, optimizer = engine.setup(model, optimizer)
    dataloader = engine.setup_dataloaders(dataloader)

    for batch in dataloader:
        # Your custom update logic
        loss = update_step(model, batch)

        # Logging
        if engine.is_global_zero:
             log(loss)
