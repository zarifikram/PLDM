DATA_PATH=/data/zikram/PLDM/waldl

mkdir -p $DATA_PATH/rendered

for path in $DATA_PATH/*.npz; do
    echo "Rendering $path"
    output_path=$DATA_PATH/rendered/$(basename $path)
    # remove no_images from the name
    output_path=${output_path//_no_images/}
    echo "running rendering for $path, saving to $output_path"
    python render_images.py --input_path $path --output_path ${output_path} --config configs/good_quality_data.yaml
done

/data/zikram/PLDM/wall/noise_mix_0.16_no_images.npz  
/data/zikram/PLDM/wall/random_trajectories_no_images.npz
/data/zikram/PLDM/wall/rendered/noise_mix_0.08.npz 
 /data/zikram/PLDM/wall/rendered/noise_mix_0.04.npz 
 /data/zikram/PLDM/wall/rendered/noise_mix_0.02.npz   
 /data/zikram/PLDM/wall/rendered/noise_mix_0.01.npz   
 /data/zikram/PLDM/wall/rendered/noise_mix_0.001.npz      