from preprocess import generate_
#dp_micro_num = [5,3] falcon
recomp_stages = []
recomp_microbatches = []
recomp_layers = []
open_recomp = False
num_microbatches, pp_size, stage_placement, placement_strategy, \
split_backward, unified_scheduler, comm_graph, first_stage, \
last_stage, Devices_containing_last_stage, recomp_stages, all_pre_fetch_w = generate_() #recomp_layers,recomp_microbatches

if len(recomp_stages) > 0:
    open_recomp = True
layerwise = False

#straggler congig
#slow_comm
add_president_latency = False
comm_latency_pair =[(2,3),(3,2)] #pp_rank_pair
latency_time = 10 #ms
add_random_latency = False
latency_step_index = [[] for _ in range(pp_size)]
latency_step_index[2] = [13]
latency_step_index[3] = [20]
#solw_compute
add_slow_compute = True
slow_compute_rank = None
slow_microbatch_id = None
slow_compute_time = [10,20]#ms
#alpa or heter config
heter = False
alpa = False
# layer_placement = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14], [15, 16]]
#metis[[1,6],[7,11],[12,13],[14,16]] [[1, 3], [4, 5], [6, 8], [9, 11], [12, 12], [13, 13], [14, 14], [15, 16]]
#H800 profile
sleep_forward_time = 0
sleep_backward_time = 0
sleep_forward_time_perlayer = 3
sleep_backward_time_perlayer = 4
# if tp_size == 2:
#     sleep_forward_time_perlayer = 8
#     sleep_backward_time_perlayer = 16
# elif tp_size == 4:
#     sleep_forward_time_perlayer = 6
#     sleep_backward_time_perlayer = 12
# elif tp_size == 8:
#     sleep_forward_time_perlayer = 4
#     sleep_backward_time_perlayer = 8
# if not alpa:
#     if pp_mode == "zbv":
#         num_chunks = 2
#     elif pp_mode == "zbh1":
#         num_chunks = 1
#     layers_one_chunk = NUM_LAYER//pp_size//num_chunks
#     sleep_forward_time = layers_one_chunk*sleep_forward_time_perlayer
#     sleep_backward_time = layers_one_chunk*sleep_backward_time_perlayer
#     print(f"sleep_forward_time:{sleep_forward_time}, sleep_backward_time:{sleep_backward_time}")

