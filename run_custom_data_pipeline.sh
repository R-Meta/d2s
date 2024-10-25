workspace_path=$1

python run_custom_data/pipeline.py $workspace_path

echo "Finished creating SfM model for custom data"

python runners/train.py --dataset_dir $workspace_path --sfm_dir $workspace_path/sfm_superpoint+superglue/ --dataset custom --scene custom

python runners/eval.py --dataset_dir $workspace_path --sfm_dir $workspace_path/sfm_superpoint+superglue/ --dataset custom --scene custom
