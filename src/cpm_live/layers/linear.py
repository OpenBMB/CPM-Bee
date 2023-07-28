# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import bmtrain as bmt
import math
import torch.nn.functional as F
import bitsandbytes as bnb
from typing import TypeVar,overload,Optional,Union,Callable,Any
from torch import Tensor, device, dtype
from bmtrain.utils import round_up
from bmtrain.global_var import config
T = TypeVar("T", bound="torch.nn.Module")

class Linear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale_before: bool = False,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale_before = scale_before

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(
                torch.nn.init.normal_, mean=init_mean, std=init_std
            ),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale_before:
            x = x / math.sqrt(self.dim_in)
            x = F.linear(x, self.weight)
        else:
            x = F.linear(x, self.weight)
            x = x / math.sqrt(self.dim_in)
        return x

class Linear4bit(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        compute_dtype: torch.dtype = torch.float32,
        compress_statistics: bool = True,
        quant_type: str = 'nf4',
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out

        weight = Params4bit(
            data=torch.empty((dim_out * dim_in // 2, 1), dtype=torch.uint8),
            requires_grad=False,
            compress_statistics=compress_statistics,
            quant_type=quant_type,   
        )
        
        self.weight = DistributedParameter4Int8(weight, requires_grad=False, quant_state=weight.quant_state)
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor):
        if getattr(self.weight, 'quant_state', None) is None:
            print('quantization state not initialized. Please ensure that the model parameters you load include the quant_state attribute.')
        
        inp_dtype = x.dtype
        dtype_dict = {
            'torch.float32': torch.float32,
            'torch.float16': torch.float16,
        }
        if self.compute_dtype is not None:
            if isinstance(self.compute_dtype, str):
                self.compute_dtype = dtype_dict[self.compute_dtype]
            x = x.to(dtype=self.compute_dtype)

        out = bnb.matmul_4bit(x, self.weight.t(), bias=None, quant_state=self.weight.quant_state)
        out = out.to(inp_dtype)
        out = out / math.sqrt(self.dim_in)
        return out

class Params4bit(torch.nn.Parameter):
    def __new__(cls,
                data=None, 
                requires_grad=True, 
                quant_state=None, 
                blocksize=64, 
                compress_statistics=True, 
                quant_type='nf4',
            ):
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.data = data
        return self
    
    def cuda(self, device):
        w = self.data.contiguous().half().cuda(device)
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type)
        self.data = w_4bit
        self.quant_state = quant_state
        return self

    @overload
    def to(self: T, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ..., non_blocking: bool = ...,) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if (device is not None and device.type == "cuda" and self.data.device.type == "cpu"):
            return self.cuda(device)
        else:
            s = self.quant_state
            if s is not None:
                # make sure the quantization state is on the right device
                s[0] = s[0].to(device)
                if self.compress_statistics:
                    # TODO: refactor this. This is a nightmare
                    # for 4-bit: 
                    # state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
                    # state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
                    #s[-2][0] = s[-2][0].to(device) # offset
                    #s[-2][1][0] = s[-2][1][0].to(device) # nested absmax

                    # for 8-bit
                    s[-2][0] = s[-2][0].to(device) # offset
                    s[-2][1][0] = s[-2][1][0].to(device) # nested quantiation state statitics
                    s[-2][1][1] = s[-2][1][1].to(device) # nested quantiation codebook
            new_param = Params4bit(super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                                  requires_grad=self.requires_grad, quant_state=self.quant_state,
                                   blocksize=self.blocksize, compress_statistics=self.compress_statistics,
                                   quant_type=self.quant_type)

            return new_param

class DistributedParameter4Int8(bmt.DistributedParameter):
    r"""
    DistributedParameter4Int8 is a subclass of DistributedParameter.
    
    The main difference is the added support for quantization, provided by the quant_state attribute.

    Args:
        data (Tensor): Parameter tensor.
        requires_grad (bool, optional): If the parameter requires gradient.
        init_method (Callable[['DistributedParameter'], None], optional): The method to initialize the parameter.
        group (str, optional): The group name of the parameter.
        quant_state (Any, optional): The state of quantization for the parameter.

    Note: DistributedParameter4Int8 must be on the CUDA device. It will transfer the data to device automatically when `__init__` called.
    """
    
    _original_shape : torch.Size
    _start_partition : int
    _end_partition : int
    _init_method : Optional[Callable[['DistributedParameter'], None]]
    _in_checkpoint_block : bool
    _group : Optional[str]
    _quant_state : Optional[Any]

    def __new__(cls,
            data : torch.Tensor, 
            requires_grad : bool = True, 
            init_method : Optional[Callable[['DistributedParameter'], None]] = None,
            group : Optional[str] = None,
            quant_state : Optional[Any] = None
        ):
        if not config["initialized"]:
            raise RuntimeError("BMTrain is not initialized")

        num_of_elements = data.numel()

        cuda_tensor = torch.tensor([], dtype=data.dtype, device="cuda") 
        cuda_storage_size = round_up(num_of_elements, config["world_size"]) // config["world_size"]

        original_shape = data.size()

        cuda_storage = cuda_tensor.storage_type()(cuda_storage_size)

        start_of_partition = cuda_storage_size * config["rank"]
        end_of_partition = min(num_of_elements, cuda_storage_size * (config["rank"] + 1))

        # FX: cuda_tensor_size < 0 if num_of_elements is too small
        cuda_tensor_size = max(end_of_partition - start_of_partition, 0)

        cuda_tensor.set_(cuda_storage, 0, (cuda_tensor_size,))
        cuda_tensor.copy_(data.view(-1)[start_of_partition: end_of_partition])
        ret = torch.Tensor._make_subclass(cls, cuda_tensor, requires_grad)
        
        setattr(ret, "_original_shape", original_shape)
        setattr(ret, "_start_partition", start_of_partition)
        setattr(ret, "_end_partition", end_of_partition)
        setattr(ret, "_init_method", init_method)
        setattr(ret, "_in_checkpoint_block", False)
        setattr(ret, "_group", group)
        setattr(ret, "_quant_state", quant_state)
        
        return ret

    @property
    def quant_state(self) -> Optional[Any]:
        return self._quant_state
    
    @quant_state.setter
    def quant_state(self, value: Optional[Any]): 
        self._quant_state = value
