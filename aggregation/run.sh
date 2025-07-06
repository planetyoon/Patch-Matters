export HF_HOME=$HOME/.cache
export TRANSFORMERS_CACHE=$HOME/.cache
export XDG_CACHE_HOME=$HOME/.cache

set -ex

# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
node_index=0
node_num=1 
chunk_num=8


for (( chunk_index=0; chunk_index<$chunk_num; chunk_index++ ))
do
    # Assign each process to a unique GPU using chunk_index
    gpu_id=$chunk_index  # Assuming chunk_index corresponds to a GPU (e.g., 0, 1, 2 for 3 GPUs)
    
    # Set CUDA_VISIBLE_DEVICES to assign a unique GPU to each process
    CUDA_VISIBLE_DEVICES=0 python3 /root/Patch-Matters/aggregation/main.py \
        --input_data /root/Patch-Matters/description_generate/description_output.json \
        --output_folder /root/Patch-Matters/aggregation/ \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num
done

wait

python3 /root/Patch-Matters/aggregation/combine.py \
    --folder_path /root/Patch-Matters/aggregation \
    --output_file /root/Patch-Matters/aggregation/aggregation_output.json
