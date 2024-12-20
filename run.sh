# CONSOLE_LOG_LEVEL="INFO" 
torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0   main.py