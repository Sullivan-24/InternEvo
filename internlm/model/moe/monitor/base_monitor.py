# Copyright (c) InternLM. All rights reserved.
from typing import Callable, Dict

import torch

from internlm.accelerator import get_accelerator
from internlm.utils.logger import get_logger

internlm_accelerator = get_accelerator()
logger = get_logger(__file__)

uniform_map: Dict[torch.device, Callable] = {}


class BaseMonitor:
    """
    Monitor base class for monitoring MoE experts.
    """

    def __init__(self):
        """
        Initializes the BaseMonitor with a common configuration for monitoring.

        """
        self.handles = []

    def clear_handles(self):
        """Clear all asynchronous handles."""
        for handle in self.handles:
            handle.wait()

        self.handles = []
