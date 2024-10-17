import argparse
import os
import pathlib
import pytest
import sys
import typing

import determined as det
from determined import pytorch

from tests.experiment import deepspeed_utils, utils  # noqa: I100
from tests.experiment.fixtures import deepspeed_linear_model


@pytest.mark.pytorch
class TestDeepSpeedTrial:
    def setup_method(self) -> None:
        # This training setup is not guaranteed to converge in general,
        # but has been tested with this random seed.  If changing this
        # random seed, verify the initial conditions converge.
        self.trial_seed = 17
        self.hparams = {
            "hidden_size": 2,
            "learning_rate": 0.5,
            "global_batch_size": 4,
            "dataloader_type": "determined",
            "disable_dataset_reproducibility_checks": False,
        }
    def test_deepspeed_linear(self, tmp_path: pathlib.Path) -> None:
        def linear_trial_loss_validation(tmp_path: pathlib.Path):
            """Assert that the training loss and validation error decrease monotonically."""
            tensorboard_path = tmp_path.joinpath("tensorboard")
            trial, trial_controller = deepspeed_utils.create_trial_and_trial_controller(
                trial_class=deepspeed_linear_model.LinearDeepSpeedTrial,
                hparams=self.hparams,
                trial_seed=self.trial_seed,
                tensorboard_path=tensorboard_path,
            )

            train_steps, metrics = trial_controller._train_with_boundaries(
                training_enumerator=enumerate(trial_controller.training_iterator),
                train_boundaries=[
                    pytorch._TrainBoundary(
                        step_type=pytorch.TrainBoundaryType.TRAIN, unit=pytorch.Batch(100)
                    )
                ],
            )

            assert len(train_steps) == 1, "unexpected train step count"
            assert train_steps[0].limit_reached, "train step did not reach expected limit"
            assert len(metrics) == 100, "metrics length did not match input"

            for older, newer in zip(metrics, metrics[1:]):
                assert newer["loss"] <= older["loss"]

    def test_checkpointing_and_restoring(self, tmp_path: pathlib.Path) -> None:
        self.checkpoint_and_check_metrics(
            deepspeed_linear_model.LinearDeepSpeedTrial, tmp_path, (100, 100)
        )
    def checkpoint_and_check_metrics(
        self,
        trial_class: deepspeed_linear_model.LinearDeepSpeedTrial,
        hparams: typing.Dict,
        tmp_path: pathlib.Path,
        steps: typing.Tuple[int, int] = (1, 1),
    ) -> typing.Tuple[
        typing.Sequence[typing.Dict[str, typing.Any]], typing.Sequence[typing.Dict[str, typing.Any]]
    ]:
        checkpoint_dir = str(tmp_path.joinpath("checkpoint"))
        tensorboard_path = tmp_path.joinpath("tensorboard")
        training_metrics = {"A": [], "B": []}
        validation_metrics = {"A": [], "B": []}

        # Trial A: train some batches and checkpoint
        trial_A, trial_controller_A = deepspeed_utils.create_trial_and_trial_controller(
            trial_class=trial_class,
            hparams=hparams,
            trial_seed=self.trial_seed,
            max_batches=steps[0],
            min_validation_batches=steps[0],
            min_checkpoint_batches=steps[0],
            checkpoint_dir=checkpoint_dir,
            tensorboard_path=tensorboard_path,
        )

        trial_controller_A.run()

        metrics_callback = trial_A.metrics_callback
        checkpoint_callback = trial_A.checkpoint_callback

        training_metrics["A"] = metrics_callback.training_metrics
        assert (
            len(training_metrics["A"]) == steps[0]
        ), "training metrics did not match expected length"
        validation_metrics["A"] = metrics_callback.validation_metrics

        assert len(checkpoint_callback.uuids) == 1, "trial did not return a checkpoint UUID"

        # Trial A: restore from checkpoint and train
        trial_A, trial_controller_A = deepspeed_utils.create_trial_and_trial_controller(
            trial_class=trial_class,
            hparams=hparams,
            trial_seed=self.trial_seed,
            max_batches=steps[0] + steps[1],
            min_validation_batches=steps[1],
            min_checkpoint_batches=sys.maxsize,
            checkpoint_dir=checkpoint_dir,
            tensorboard_path=tensorboard_path,
            latest_checkpoint=checkpoint_callback.uuids[0],
            steps_completed=trial_controller_A.state.batches_trained,
        )
        trial_controller_A.run()

        metrics_callback = trial_A.metrics_callback
        training_metrics["A"] += metrics_callback.training_metrics
        validation_metrics["A"] += metrics_callback.validation_metrics

        assert (
            len(training_metrics["A"]) == steps[0] + steps[1]
        ), "training metrics returned did not match expected length"

        # Trial B: run for some steps
        trial_B, trial_controller_B = deepspeed_utils.create_trial_and_trial_controller(
            trial_class=trial_class,
            hparams=hparams,
            trial_seed=self.trial_seed,
            max_batches=steps[0] + steps[1],
            min_validation_batches=steps[0],
            min_checkpoint_batches=sys.maxsize,
            checkpoint_dir=checkpoint_dir,
            tensorboard_path=tensorboard_path,
        )
        trial_controller_B.run()

        metrics_callback = trial_B.metrics_callback

        training_metrics["B"] = metrics_callback.training_metrics
        validation_metrics["B"] = metrics_callback.validation_metrics

        for A, B in zip(training_metrics["A"], training_metrics["B"]):
            utils.assert_equivalent_metrics(A, B)

        for A, B in zip(validation_metrics["A"], validation_metrics["B"]):
            utils.assert_equivalent_metrics(A, B)

        return (training_metrics["A"], training_metrics["B"])
