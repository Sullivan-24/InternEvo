from .mtypes import *
device_size = 3             # Size of devices.
pp_size = device_size * 2   # Size of pipeline.
model_size = pp_size + 0    # Model layers, should be equal or larger than pipeline size.
nvs = (1,)                  # Number of virtual stages / number of stages.
nmb = device_size + 0       # Number of microbatch.
time_limit = 60
stage_order_search = StageSearchOrder.Random

ft_bt_rate = 1.80
ft_wt_rate = 0.60      
ft = 20                 # Forward pass time.
bt = ft * ft_bt_rate    # Backpropagation time for compute the gradients of layer inputs.
wt = ft * ft_wt_rate    # Backpropagation time for compute the gradients of layer parameters.
comm = 1

def set_execution_time(time_value, 
                       stages_num = None, 
                       num_microbatch = None, 
                       comm_time = None):
    global ft, bt, wt, pp_size, model_size, nmb, comm
    ft = time_value
    bt = ft * ft_bt_rate
    wt = ft  
    pp_size = stages_num if stages_num else pp_size
    model_size = pp_size
    nmb = num_microbatch if num_microbatch else pp_size * 1.0
    comm = comm_time if comm_time else comm