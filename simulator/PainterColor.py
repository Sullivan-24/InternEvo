from .abstract.mutils import *
def set_color(sid, workload_type, layer_num, layer_wise = LAYERWISE):
    color = None
    if workload_type == 'f':    #颜色设置，加上w的情况
        color = "#FFF2CC"
    elif workload_type == 'b':
        color = "#DAE8FC" 
    else:
        color = "#D5E8D4"

    if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE or layer_wise:
        if sid == 0:
            color = "#F8CECC" # 红色: #F8CECC
        elif sid == layer_num - 2:
            if workload_type == 'f':
                color = "#E1D5E7" # 紫色: #E1D5E7
            elif workload_type == 'b':
                color = "#E1D5E7"
            else:
                color = "#E1D5E7"
        elif sid == layer_num - 1:
            if workload_type == 'f':
                color = "#FFE6CC" # 橙色: #FFE6CC
            elif workload_type == 'b':
                color = "#FFE6CC"
            else:
                color = "#FFE6CC"
    return color