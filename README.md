# Free Lunch

You can install the required dependencies in two steps:

1. CD to `trident-xtreme`
2. `conda env create -f environment.yaml`
3. Activate the conda environment `conda env activate trident_xtreme`
4. Change your working directory to `trident`
5. `pip install -e ./`

Then switch to `trident-xtreme` and

1. `conda activate trident_xtreme`
2. `bash $YOUR_EXPERIMENT.sh`
