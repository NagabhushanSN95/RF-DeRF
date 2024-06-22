#!/bin/bash
# Shree KRISHNAya Namaha

scene_names=(
        "Birthday"
        "Painter"
        "Remy"
        "Theater"
        "Train"
        )

# Loop over every scene and run training and rendering
train_dirpath=$(dirname $1)
for scene_name in ${scene_names[@]};
do
    # Create the config file for the scene
    python utils/ConfigsCreator.py --configs-path $1 --scene-names $scene_name

    scene_configs_path="$train_dirpath/$scene_name/dynerf.py"

    # Call training
    python main.py --config-path $scene_configs_path

    # Call validation
    # python main.py --config-path $scene_configs_path --validate-only

    # Call validation on train set
    # python main.py --config-path $scene_configs_path --validate-train-only

    # Call rendering
    python main.py --config-path $scene_configs_path --render-only
done
