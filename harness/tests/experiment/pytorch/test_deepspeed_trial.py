import pathlib
import pytest

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
                    step_type=pytorch._TrainBoundaryType.TRAIN, unit=pytorch.Batch(100)
                )
            ],
        )

        assert len(train_steps) == 1, "unexpected train step count"
        assert train_steps[0].limit_reached, "train step did not reach expected limit"
        assert len(metrics) == 100, "metrics length did not match input"

        for older, newer in zip(metrics, metrics[1:]):
            assert newer["loss"] <= older["loss"]

    def test_training_metrics(self, tmp_path: pathlib.Path) -> None:
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
                    step_type=pytorch._TrainBoundaryType.TRAIN, unit=pytorch.Batch(100)
                )
            ],
        )

        assert len(train_steps) == 1, "unexpected train step count"
        assert train_steps[0].limit_reached, "train step did not reach expected limit"
        assert len(metrics) == 100, "metrics length did not match input"
        for metric in metrics:
            assert "mse" in metric

    def test_checkpointing_and_restoring(self, tmp_path: pathlib.Path) -> None:
        updated_hparams = {
            "lr_scheduler_step_mode": pytorch.LRScheduler.StepMode.STEP_EVERY_BATCH.value,
            **self.hparams,
        }
        self.checkpoint_and_check_metrics(
            deepspeed_linear_model.LinearDeepSpeedTrial, updated_hparams, tmp_path, (100, 100)
        )
