"""Pinned-SLIME training driver with post-update weight verification."""

from __future__ import annotations

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action


def train(args) -> None:
    configure_logger()
    placement_groups = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, placement_groups["rollout"])
    actor_model, critic_model = create_training_models(args, placement_groups, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())
    actor_model.update_weights()
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))
        print("Verified initial actor-to-SGLang weight synchronization", flush=True)
    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    def offload_train(rollout_id: int) -> None:
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id: int) -> None:
        if not args.use_critic or rollout_id >= args.num_critic_only_steps:
            actor_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
        if args.use_critic:
            critic_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))
            print(f"Verified post-update actor-to-SGLang weights for rollout {rollout_id}", flush=True)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    train(parse_args())
