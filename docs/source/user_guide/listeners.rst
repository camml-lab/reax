Listeners
=========

Listeners allow you to customise the behaviour of the training loop at specific points, such as the
start of an epoch or after a batch.

Built-in Listeners
------------------

REAX comes with several built-in listeners:

*   :class:`~reax.listeners.ModelCheckpoint`: Automatically save model checkpoints based on a
    monitored metric.
*   :class:`~reax.listeners.EarlyStopping`: Stop training early if a metric stops improving.
*   :class:`~reax.listeners.ProgressBar`: Display a progress bar during training.

Custom Listeners
----------------

You can create your own listener by inheriting from :class:`~reax.TrainerListener`.

.. code-block:: python

    import reax

    class MyPrintingListener(reax.TrainerListener):
        def on_train_start(self, trainer, module):
            print("Training is starting!")

        def on_train_end(self, trainer, module):
            print("Training is ending!")

Using Listeners
---------------

Pass your listeners to the Trainer:

.. code-block:: python

    trainer = reax.Trainer(listeners=[MyPrintingListener()])

Order of Execution
------------------

Listeners are executed in the order they are passed to the Trainer.
