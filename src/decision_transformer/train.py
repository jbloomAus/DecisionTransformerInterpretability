import pytest
import torch as t
import torch.nn as nn
from einops import rearrange
from dataclasses import asdict
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

import wandb
from src.config import EnvironmentConfig, OfflineTrainConfig
from src.models.trajectory_transformer import (
    CloneTransformer,
    DecisionTransformer,
    TrajectoryTransformer,
)

from .offline_dataset import TrajectoryDataset
from .eval import evaluate_dt_agent
from .utils import configure_optimizers, get_scheduler, store_model_checkpoint


def train(
    model: TrajectoryTransformer,
    trajectory_data_set: TrajectoryDataset,
    env,
    make_env,
    offline_config: OfflineTrainConfig,
    device="cpu",
    track=False,
    exp_name=""
):
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)

    train_dataloader, test_dataloader = get_dataloaders(
        trajectory_data_set, offline_config
    )

    if track:
        checkpoint_artifact = wandb.Artifact(
            f"{exp_name}_checkpoints", type="model"
        )
        checkpoint_num = 1
        checkpoint_interval = offline_config.train_epochs // offline_config.num_checkpoints + 1
        checkpoint_num = store_model_checkpoint(
            model,
            exp_name,
            offline_config,
            checkpoint_num,
            checkpoint_artifact
        )

    # get optimizer from string
    optimizer = configure_optimizers(model, offline_config)
    # TODO: Stop passing through all the args to the scheduler, shouldn't be necessary.
    scheduler_config = asdict(offline_config)
    del scheduler_config["optimizer"]
    # get total number of training steps.
    train_batches_per_epoch = len(train_dataloader)
    scheduler_config["training_steps"] = (
        offline_config.train_epochs * train_batches_per_epoch
    )
    scheduler = get_scheduler(
        offline_config.scheduler, optimizer, **scheduler_config
    )
    # can uncomment this to get logs of gradients and pars.
    # wandb.watch(model, log="all", log_freq=train_batches_per_epoch)
    pbar = tqdm(range(offline_config.train_epochs))
    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m) in enumerate(train_dataloader):
            total_batches = epoch * train_batches_per_epoch + batch

            model.train()

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n  # dummy action for padding

            optimizer.zero_grad()

            if isinstance(model, DecisionTransformer):
                action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None
                _, action_preds, _ = model(
                    states=s,
                    # remove last action
                    actions=action,
                    rtgs=rtg,  # remove last rtg
                    timesteps=ti.unsqueeze(-1),
                )
            elif isinstance(model, CloneTransformer):
                _, action_preds = model(
                    states=s,
                    # remove last action
                    actions=a[:, :-1].unsqueeze(-1)
                    if a.shape[1] > 1
                    else None,
                    timesteps=ti.unsqueeze(-1),
                )

            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            # ignore dummy action
            loss = loss_fn(
                action_preds[a_exp != env.action_space.n],
                a_exp[a_exp != env.action_space.n],
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Training DT: {loss.item():.4f}")

            if offline_config.track:
                tokens_seen = (
                    (total_batches + 1)
                    * offline_config.batch_size
                    * model.transformer_config.n_ctx
                )
                learning_rate = optimizer.param_groups[0]["lr"]
                wandb.log({"train/loss": loss.item()}, step=total_batches)
                wandb.log(
                    {"metrics/tokens_seen": tokens_seen}, step=total_batches
                )
                wandb.log(
                    {"metrics/learning_rate": learning_rate},
                    step=total_batches,
                )

        # # at test frequency
        if epoch % offline_config.test_frequency == 0:
            test(
                model=model,
                dataloader=test_dataloader,
                env=env,
                epochs=offline_config.test_epochs,
                track=offline_config.track,
                batch_number=total_batches,
            )

        eval_env_config = EnvironmentConfig(
            env_id=env.spec.id,
            capture_video=True,
            max_steps=min(
                model.environment_config.max_steps,
                offline_config.eval_max_time_steps,
            ),
            fully_observed=False,
            one_hot_obs=(trajectory_data_set.observation_type == "one_hot"),
            view_size=env.observation_space["image"].shape[0]
            if "image" in list(env.observation_space.keys())
            else 7,
        )

        eval_env_func = make_env(
            config=eval_env_config,
            seed=batch,
            idx=0,
            run_name=f"dt_eval_videos_{batch}",
        )

        if epoch % offline_config.eval_frequency == 0:
            for rtg in offline_config.initial_rtg:
                evaluate_dt_agent(
                    env_id=env.spec.id,
                    model=model,
                    env_func=eval_env_func,
                    trajectories=offline_config.eval_episodes,
                    track=offline_config.track,
                    batch_number=total_batches,
                    initial_rtg=float(rtg),
                    device=device,
                    num_envs=offline_config.eval_num_envs,
                )

        if track and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_num = store_model_checkpoint(
                model,
                exp_name,
                offline_config,
                checkpoint_num,
                checkpoint_artifact
            )

    if track:
        store_model_checkpoint(
            model,
            exp_name,
            offline_config,
            checkpoint_num,
            checkpoint_artifact
        )
        wandb.log_artifact(checkpoint_artifact)

    return model


@pytest.mark.skip(reason="This is not a test")
def test(
    model: TrajectoryTransformer,
    dataloader: DataLoader,
    env,
    epochs=10,
    track=False,
    batch_number=0,
):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss = 0
    n_correct = 0
    n_actions = 0

    pbar = tqdm(range(epochs))
    test_batches_per_epoch = len(dataloader)

    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m) in enumerate(dataloader):
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n

            if isinstance(model, DecisionTransformer):
                _, action_preds, _ = model(
                    states=s,
                    actions=a[:, :-1].unsqueeze(-1)
                    if a.shape[1] > 1
                    else None,
                    rtgs=rtg,
                    timesteps=ti.unsqueeze(-1),
                )
            elif isinstance(model, CloneTransformer):
                _, action_preds = model(
                    states=s,
                    # remove last action
                    actions=a[:, :-1].unsqueeze(-1)
                    if a.shape[1] > 1
                    else None,
                    timesteps=ti.unsqueeze(-1),
                )

            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            a_hat = t.argmax(action_preds, dim=-1)
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            action_preds = action_preds[a_exp != env.action_space.n]
            a_hat = a_hat[a_exp != env.action_space.n]
            a_exp = a_exp[a_exp != env.action_space.n]

            n_actions += a_exp.shape[0]
            n_correct += (a_hat == a_exp).sum()
            loss += loss_fn(action_preds, a_exp)

            accuracy = n_correct.item() / n_actions
            pbar.set_description(f"Testing DT: Accuracy so far {accuracy:.4f}")

    mean_loss = loss.item() / (epochs * test_batches_per_epoch)

    if track:
        wandb.log({"test/loss": mean_loss}, step=batch_number)
        wandb.log({"test/accuracy": accuracy}, step=batch_number)

    return mean_loss, accuracy


def get_dataloaders(trajectory_data_set, offline_config):
    train_dataset, test_dataset = random_split(
        trajectory_data_set, [0.90, 0.10]
    )

    # Create the train DataLoader
    train_sampler = WeightedRandomSampler(
        weights=trajectory_data_set.sampling_probabilities[
            train_dataset.indices
        ],
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=offline_config.batch_size,
        sampler=train_sampler,
    )

    # Create the test DataLoader
    test_sampler = WeightedRandomSampler(
        weights=trajectory_data_set.sampling_probabilities[
            test_dataset.indices
        ],
        num_samples=len(test_dataset),
        replacement=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=offline_config.batch_size,
        sampler=test_sampler,
    )

    return train_dataloader, test_dataloader