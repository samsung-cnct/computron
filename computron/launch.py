from dataclasses import dataclass, field
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from energonai import BatchManager, SubmitEntry
from pydantic import BaseModel
import torch.multiprocessing as mp

from computron.controller import Controller, LRUController
from computron.engine import Engine
from computron.worker import Worker


@dataclass
class ModelConfig:
    model_id: str
    master_host: str
    master_port: int
    rpc_port: int
    request_port: int
    request_type: BaseModel
    unpack_request_fn: Callable[[BaseModel], SubmitEntry]
    pack_response_fn: Callable[[Any], BaseModel]
    model_fn: Callable
    pipelinable: bool = False
    batch_manager: Optional[BatchManager] = None
    pipe_size: int = 1
    queue_size: int = 0
    rpc_disable_shm: bool = True
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class LogWrapper:
    def __init__(self, task: Callable, log_file: str):
        self.task = task
        self.log_file = log_file

    def __call__(self, *args, **kwargs):
        with open(self.log_file, "w+") as f:
            sys.stdout = f
            self.task(*args, **kwargs)


def _launch_offloading_workers(
    config: ModelConfig,
    tp_world_size: int,
    pp_world_size: int,
    n_proc_per_node: int = 1,
    node_rank: int = 0,
    log_dir: Optional[str] = None,
):
    ctx = mp.get_context("spawn")
    procs = []
    for i in range(n_proc_per_node):
        rank = n_proc_per_node * node_rank + i
        if log_dir is None:
            target = Worker
        else:
            log_file = os.path.join(log_dir, f"{config.model_id}_w{i}.log")
            target = LogWrapper(Worker, log_file)
        p = ctx.Process(
            target=target,
            args=(
                rank,
                tp_world_size,
                pp_world_size,
                config.master_host,
                config.master_port,
                config.rpc_port,
                n_proc_per_node,
                config.model_fn,
                config.pipelinable,
                config.pipe_size,
                config.rpc_disable_shm,
            ),
            kwargs=config.model_kwargs,
        )
        procs.append(p)
        p.start()


_controllers = {
    "lru": LRUController,
}


def launch_multi_model(
    model_configs: List[ModelConfig],
    tp_world_size: int,
    pp_world_size: int,
    n_nodes: int,
    node_rank: int,
    controller_type: str = "lru",
    controller_kwargs: Optional[Dict[str, Any]] = {},
    log_dir: Optional[str] = None,
) -> Optional[Controller]:
    num_models = len(model_configs)
    world_size = tp_world_size * pp_world_size
    assert world_size % n_nodes == 0
    n_proc_per_node = world_size // n_nodes

    if node_rank == 0:
        controller = _controllers[controller_type.lower()](**controller_kwargs)
    if log_dir:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    ctx = mp.get_context("spawn")
    procs = []

    for i in range(num_models):
        config = model_configs[i]
        _launch_offloading_workers(
            config,
            tp_world_size,
            pp_world_size,
            n_proc_per_node,
            node_rank,
            log_dir,
        )
        if node_rank == 0:
            if log_dir is None:
                target = Engine
            else:
                log_file = os.path.join(log_dir, f"{config.model_id}.log")
                target = LogWrapper(Engine, log_file)
            p = ctx.Process(
                target=target,
                args=(
                    tp_world_size,
                    pp_world_size,
                    config.model_id,
                    config.master_host,
                    config.rpc_port,
                    config.request_port,
                    config.request_type,
                    config.unpack_request_fn,
                    config.pack_response_fn,
                    n_proc_per_node,
                    config.batch_manager,
                    config.pipe_size,
                    config.queue_size,
                    config.rpc_disable_shm,
                ),
            )
            procs.append(p)
            p.start()

            controller.register_model(config.model_id, config.master_host, config.request_port)

    return controller

    # TODO: add signal handler that syncs with engines and workers
