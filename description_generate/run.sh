set -ex 

node_index=0
node_num=1 # Use 1 for single node, or set to the number of nodes if running on multiple node
chunk_num=8

# bash prepare.sh

for (( chunk_index=0; chunk_index<chunk_num; chunk_index++ ))
do
    # Assign each process to a unique GPU using chunk_index
    gpu_id=$chunk_index  # Assuming chunk_index corresponds to a GPU (e.g., 0, 1, 2 for 3 GPUs)
    
    # Set CUDA_VISIBLE_DEVICES to assign a unique GPU to each process
    CUDA_VISIBLE_DEVICES=0 python3 /root/Patch-Matters/description_generate/multi_process.py \
        --input_file /root/Patch-Matters/description_generate/test_data/did_bench.json \
        --output_folder /root/Patch-Matters/description_generate \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num
done

wait

python3 /root/Patch-Matters/description_generate/combine.py \
    --folder_path /root/Patch-Matters/description_generate \
    --output_file /root/Patch-Matters/description_generate/description_output.json
