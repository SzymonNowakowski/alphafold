![header](imgs/header.jpg)

# AlphaFold

This package provides an implementation of the inference pipeline of AlphaFold
v2.1.1. This is a completely new model that was entered in CASP14 and published in
Nature. For simplicity, we refer to this model as AlphaFold throughout the rest
of this document.



Any publication that discloses findings arising from using this source code or the model parameters should [cite](#citing-this-work) the
[AlphaFold  paper](https://doi.org/10.1038/s41586-021-03819-2) and, if
applicable, the [AlphaFold-Multimer paper](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1).

Please also refer to the
[Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf)
for a detailed description of the method.

![CASP14 predictions](imgs/casp14_predictions.gif)

## Center4ML fork
This package was forked from the original [DeepMind github](https://github.com/deepmind/alphafold) and is under development by
[Center4ML Team](https://center4ml.idub.uw.edu.pl/) to provide the following two changes:

* the option to run the CPU-intensive database search pipeline **separately** from the GPU-intensive 
   neural network inference and
* the option to start folding from a researcher-provided structural hypothesis.

## First time setup

The following steps are required in order to run AlphaFold:

1.  Install [Docker](https://www.docker.com/).
    *   Install
        [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
        for GPU support.
    *   Setup running
        [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).
1.  Download genetic databases (see below).
1.  Download model parameters (see below).
1.  Check that AlphaFold will be able to use a GPU by running:

    ```bash
    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
    ```

    The output of this command should show a list of your GPUs. If it doesn't,
    check if you followed all steps correctly when setting up the
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    or take a look at the following
    [NVIDIA Docker issue](https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-801479573).

If you wish to run AlphaFold using Singularity (a common containerization platform on HPC systems) we recommend using some of the
third party Singularity setups as linked in
https://github.com/deepmind/alphafold/issues/10 or
https://github.com/deepmind/alphafold/issues/24.

### Genetic databases

This step requires `aria2c` to be installed on your machine.

AlphaFold needs multiple genetic (sequence) databases to run:

*   [BFD](https://bfd.mmseqs.com/),
*   [MGnify](https://www.ebi.ac.uk/metagenomics/),
*   [PDB70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/),
*   [PDB](https://www.rcsb.org/) (structures in the mmCIF format),
*   [PDB seqres](https://www.rcsb.org/) – only for AlphaFold-Multimer,
*   [Uniclust30](https://uniclust.mmseqs.com/),
*   [UniProt](https://www.uniprot.org/uniprot/) – only for AlphaFold-Multimer,
*   [UniRef90](https://www.uniprot.org/help/uniref).

We provide a script `scripts/download_all_data.sh` that can be used to download
and set up all of these databases:

*   Default:

    ```bash
    scripts/download_all_data.sh <DOWNLOAD_DIR>
    ```

    will download the full databases.

*   With `reduced_dbs`:

    ```bash
    scripts/download_all_data.sh <DOWNLOAD_DIR> reduced_dbs
    ```

    will download a reduced version of the databases to be used with the
    `reduced_dbs` database preset.

:ledger: **Note: The download directory `<DOWNLOAD_DIR>` should _not_ be a
subdirectory in the AlphaFold repository directory.** If it is, the Docker build
will be slow as the large databases will be copied during the image creation.

We don't provide exactly the database versions used in CASP14 – see the [note on
reproducibility](#note-on-reproducibility). Some of the databases are mirrored
for speed, see [mirrored databases](#mirrored-databases).

:ledger: **Note: The total download size for the full databases is around 415 GB
and the total size when unzipped is 2.2 TB. Please make sure you have a large
enough hard drive space, bandwidth and time to download. We recommend using an
SSD for better genetic search performance.**

The `download_all_data.sh` script will also download the model parameter files.
Once the script has finished, you should have the following directory structure:

```
$DOWNLOAD_DIR/                             # Total: ~ 2.2 TB (download: 438 GB)
    bfd/                                   # ~ 1.7 TB (download: 271.6 GB)
        # 6 files.
    mgnify/                                # ~ 64 GB (download: 32.9 GB)
        mgy_clusters_2018_12.fa
    params/                                # ~ 3.5 GB (download: 3.5 GB)
        # 5 CASP14 models,
        # 5 pTM models,
        # 5 AlphaFold-Multimer models,
        # LICENSE,
        # = 16 files.
    pdb70/                                 # ~ 56 GB (download: 19.5 GB)
        # 9 files.
    pdb_mmcif/                             # ~ 206 GB (download: 46 GB)
        mmcif_files/
            # About 180,000 .cif files.
        obsolete.dat
    pdb_seqres/                            # ~ 0.2 GB (download: 0.2 GB)
        pdb_seqres.txt
    small_bfd/                             # ~ 17 GB (download: 9.6 GB)
        bfd-first_non_consensus_sequences.fasta
    uniclust30/                            # ~ 86 GB (download: 24.9 GB)
        uniclust30_2018_08/
            # 13 files.
    uniprot/                               # ~ 98.3 GB (download: 49 GB)
        uniprot.fasta
    uniref90/                              # ~ 58 GB (download: 29.7 GB)
        uniref90.fasta
```

`bfd/` is only downloaded if you download the full databases, and `small_bfd/`
is only downloaded if you download the reduced databases.

### Model parameters

While the AlphaFold code is licensed under the Apache 2.0 License, the AlphaFold
parameters are made available for non-commercial use only under the terms of the
CC BY-NC 4.0 license. Please see the [Disclaimer](#license-and-disclaimer) below
for more detail.

The AlphaFold parameters are available from
https://storage.googleapis.com/alphafold/alphafold_params_2021-10-27.tar, and
are downloaded as part of the `scripts/download_all_data.sh` script. This script
will download parameters for:

*   5 models which were used during CASP14, and were extensively validated for
    structure prediction quality (see Jumper et al. 2021, Suppl. Methods 1.12
    for details).
*   5 pTM models, which were fine-tuned to produce pTM (predicted TM-score) and
    (PAE) predicted aligned error values alongside their structure predictions
    (see Jumper et al. 2021, Suppl. Methods 1.9.7 for details).
*   5 AlphaFold-Multimer models that produce pTM and PAE values alongside their
    structure predictions.

### Updating existing AlphaFold installation to include AlphaFold-Multimers

If you have AlphaFold v2.0.0 or v2.0.1 you can either reinstall AlphaFold fully
from scratch (remove everything and run the setup from scratch) or you can do an
incremental update that will be significantly faster but will require a bit more
work. Make sure you follow these steps in the exact order they are listed below:

1.  **Update the code.**
    *   Go to the directory with the cloned AlphaFold repository and run
        `git fetch origin main` to get all code updates.
1.  **Download the UniProt and PDB seqres databases.**
    *   Run `scripts/download_uniprot.sh <DOWNLOAD_DIR>`.
    *   Remove `<DOWNLOAD_DIR>/pdb_mmcif`. It is needed to have PDB SeqRes and
        PDB from exactly the same date. Failure to do this step will result in
        potential errors when searching for templates when running
        AlphaFold-Multimer.
    *   Run `scripts/download_pdb_mmcif.sh <DOWNLOAD_DIR>`.
    *   Run `scripts/download_pdb_seqres.sh <DOWNLOAD_DIR>`.
1.  **Update the model parameters.**
    *   Remove the old model parameters in `<DOWNLOAD_DIR>/params`.
    *   Download new model parameters using
        `scripts/download_alphafold_params.sh <DOWNLOAD_DIR>`.
1.  **Follow [Running AlphaFold](#running-alphafold).**

#### API changes between v2.0.0 and v2.1.0

We tried to keep the API as much backwards compatible as possible, but we had to
change the following:

*   The `RunModel.predict()` now needs a `random_seed` argument as MSA sampling
    happens inside the Multimer model.
*   The `preset` flag in `run_alphafold.py` and `run_docker.py` was split into
    `db_preset` and `model_preset`.
*   The models to use are not specified using `model_names` but rather using the
    `model_preset` flag. If you want to customize which models are used for each
    preset, you will have to modify the the `MODEL_PRESETS` dictionary in
    `alphafold/model/config.py`.
*   Setting the `data_dir` flag is now needed when using `run_docker.py`.


## Running AlphaFold 

**The simplest way to run AlphaFold is using the provided Docker script.** This
was tested on Google Cloud with a machine using the `nvidia-gpu-cloud-image`
with 12 vCPUs, 85 GB of RAM, a 100 GB boot disk, the databases on an additional
3 TB disk, and an A100 GPU.

1. Clone this repository and `cd` into it.

    ```bash
    git clone https://github.com/SzymonNowakowski/alphafold.git
    ```
1. Build the Docker image:

    ```bash
    docker build -f docker/Dockerfile -t alphafold:2.1.1 .
    ```
1. Optionally, to move the image to a HPC slurm environment (customarily running Singularity, not Docker), execute the following sequence:

    ```bash
    docker image save -o local_dir/alphafold-2.1.1.docker alphafold:2.1.1
    scp local_dir/alphafold-2.1.1.docker HPC_server:remote_dir
    ssh -l user HPC_server
    user@HPC_server:~/$ singularity build remote_dir/alphafold-2.1.1.sif docker-archive://remote_dir/alphafold-2.1.1.docker
    ```
   
    **On some HPC environments, before the execution of the last step,
    it may be necessary to**

    * enter an interactive SLURM mode with a command 
    `srun -N1 -n1 --account xxxx --gres=gpu:1 --time=2-00:00:00 --pty /bin/bash -l` with `xxxx` 
   being your account or grant name (to be charged for the computations in HPC)
    * make sure that there is enough space for temporary files
      (there may be not enough space on the default temporary resource) 
      with a command `export SINGULARITY_TMPDIR=/home/$USER/tmp`
      (after making sure the `/home/$USER/tmp` directory exists and has sufficient quota)
  
   **OR, as the alternative to the last step** you may submit the following `build_singularity.slurm` script:
    ```bash
    #!/bin/bash
    #SBATCH --job-name build_alphafold2.1.1_singularity_container
    #SBATCH -A xxxx             # your account or grant name (to be charged for the computations in HPC)
    #SBATCH --time=2-00:00:00   
    #SBATCH --cpus-per-task=1   
    #SBATCH --gres=gpu:1
    mkdir /home/$USER/tmp
    export SINGULARITY_TMPDIR=/home/$USER/tmp
    singularity build remote_dir/alphafold-2.1.1.sif docker-archive://remote_dir/alphafold-2.1.1.docker
    ```
    with a command `sbatch build_singularity.slurm` 
         
1. Alternatively to the point above, you 
    may follow [Alphafold on ComputeCanada](https://docs.computecanada.ca/wiki/AlphaFold#Using_singularity)
    instructions (with the version changed) and execute:

    ```bash
    user@HPC_server:~/$ singularity build remote_dir/alphafold-2.1.1.sif docker://uvarc/alphafold:2.1.1
    ```
   
    All bolded HPC-related footnotes from the point above apply here, as well.

    Observe that [`uvarc/alphafold` link](https://hub.docker.com/r/uvarc/alphafold#!) stores 
    some other versions of the container, too. 

    Observe also, that not only the version number, but 
    the rest of [Alphafold on ComputeCanada](https://docs.computecanada.ca/wiki/AlphaFold#Using_singularity) 
    instructions also need to be updated, since the new 
    Alphafold2 v2.1.1 has some additional flags (or replaces some flags) and 
    new functionality. More on the new flags in Alphafold v2.1.1
    in Section [Running AlphaFold under HPC with Singularity](#running-alphafold-under-hpc-with-singularity) below.
    
    :ledger: **Note: This is NOT a recommended path, however, as this gives you the vanilla Alphafold v2.1.1,
    *with no Center4ML-developed features available*,
    and it is included here for completeness ot 
    this```Readme```file only.** 


1. Install the `run_docker.py` dependencies. Note: You may optionally wish to
    create a
    [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html)
    to prevent conflicts with your system's Python environment.

    ```bash
    pip3 install -r docker/requirements.txt
    ```

1. Run `run_docker.py` pointing to a FASTA file containing the protein
    sequence(s) for which you wish to predict the structure. If you are
    predicting the structure of a protein that is already in PDB and you wish to
    avoid using it as a template, then `max_template_date` must be set to be
    before the release date of the structure. You must also provide the path to
    the directory containing the downloaded databases. For example, for the
    T1050 CASP14 target:

    ```bash
    python3 docker/run_docker.py \
      --fasta_paths=T1050.fasta \
      --max_template_date=2020-05-14 \
      --data_dir=$DOWNLOAD_DIR
    ```

    By default, Alphafold will attempt to use all visible GPU devices. To use a
    subset, specify a comma-separated list of GPU UUID(s) or index(es) using the
    `--gpu_devices` flag. See
    [GPU enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration)
    for more details.

1. You can control which AlphaFold model to run by adding the
    `--model_preset=` flag. We provide the following models:

    * **monomer**: This is the original model used at CASP14 with no ensembling.

    * **monomer\_casp14**: This is the original model used at CASP14 with
      `num_ensemble=8`, matching our CASP14 configuration. This is largely
      provided for reproducibility as it is 8x more computationally
      expensive for limited accuracy gain (+0.1 average GDT gain on CASP14
      domains).

    * **monomer\_ptm**: This is the original CASP14 model fine tuned with the
      pTM head, providing a pairwise confidence measure. It is slightly less
      accurate than the normal monomer model.

    * **multimer**: This is the [AlphaFold-Multimer](#citing-this-work) model.
      To use this model, provide a multi-sequence FASTA file. In addition, the
      UniProt database should have been downloaded.

1. You can control MSA speed/quality tradeoff by adding
    `--db_preset=reduced_dbs` or `--db_preset=full_dbs` to the run command. We
    provide the following presets:

    *   **reduced\_dbs**: This preset is optimized for speed and lower hardware
        requirements. It runs with a reduced version of the BFD database.
        It requires 8 CPU cores (vCPUs), 8 GB of RAM, and 600 GB of disk space.

    *   **full\_dbs**: This runs with all genetic databases used at CASP14.

    Running the command above with the `monomer` model preset and the
    `reduced_dbs` data preset would look like this:

    ```bash
    python3 docker/run_docker.py \
      --fasta_paths=T1050.fasta \
      --max_template_date=2020-05-14 \
      --model_preset=monomer \
      --db_preset=reduced_dbs \
      --data_dir=$DOWNLOAD_DIR
    ```

### Running AlphaFold under HPC with Singularity

#### Container codebase

:ledger: **Note: The `run_alphafold.py` script
accepts the `--structural_hypothesis_file` flag as optional. In the execution guidelines below it is not provided, 
but it may be provided as well.**

Once you have completed points 1-3 from Section [Running AlphaFold](#running-alphafold)
and downloaded the databases and the model parameters, you may 
run AlphaFold under Singularity by submitting the following 
slurm job `run_alphafold.slurm` with the `FASTA` type input file located in `inputs` directory passed as a parameter:

```bash
#!/bin/bash
#SBATCH --job-name alphafold2.1.1
#SBATCH -A xxxx             # your account or grant name (to be charged for the computations in HPC)
#SBATCH --time=2-00:00:00   # or whatever fits the QoS, adjust this to match the walltime of your job
#SBATCH --cpus-per-task=8   # DO NOT INCREASE THIS AS ALPHAFOLD CANNOT TAKE ADVANTAGE OF MORE
#SBATCH --gres=gpu:1        # You need to request one GPU to be able to run AlphaFold properly
#SBATCH --mem=90G           # adjust this according to the memory requirement per node you need

###LOGGING
echo $1 >> outputs/$SLURM_JOB_ID.desc
cat $0 >> outputs/$SLURM_JOB_ID.desc

#set the environment PATH
export PYTHONNOUSERSITE=True
#module load singularity
ALPHAFOLD_DATA_PATH=remote_dir_with_protein_databases
ALPHAFOLD_MODELS=remote_dir_with_model_parameters
BASE_DIR=$(pwd)


#Run the command
singularity run --nv \
 -B $ALPHAFOLD_DATA_PATH:/data \
 -B $ALPHAFOLD_MODELS \
 --pwd  /app/alphafold remote_dir/alphafold-2.1.1.sif \
 --fasta_paths=$BASE_DIR/$1 \
 --uniref90_database_path=/data/uniref90/uniref90.fasta \
 --data_dir=/data \
 --mgnify_database_path=/data/mgnify/mgy_clusters_2018_12.fa \
 --bfd_database_path=/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
 --uniclust30_database_path=/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
 --pdb70_database_path=/data/pdb70/pdb70 \
 --template_mmcif_dir=/data/pdb_mmcif/mmcif_files \
 --obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat \
 --output_dir=$BASE_DIR/outputs \
 --model_preset=monomer \
 --db_preset=full_dbs \
 --max_template_date=2021-12-31 
```

In particular, you may submit it with a command

```bash
sbatch run_alphafold.slurm inputs/file_with_monomer.fasta
```

The results would be then written into the `outputs/file_with_monomer` directory.

#### External codebase
:ledger: **Note: The `run_alphafold_external_code.py` script
accepts the `--structural_hypothesis_file` flag as optional. In the execution guidelines below it is not provided, 
but it may be provided as well.**

For the sake of the ease of development, there is an option to run Alphafold2 from the external codebase from `alphafold_current` subdirectory.

In this setting, the singularity container provides the environment (CUDA drivers, installed packages etc.) but the code is kept externally to the docker. As an effect, you don't have to rebuild the docker every time you make *minor* code changes. If some new packages were involved in the changed code, you might have to rebuild the singularity container, too, to match the code with the installed environment.

To run it in this setting execute the following steps:

1. Execute `mkdir alphafold_current` to create the codebase subdirectory
    
1. Clone this repository into `alphafold_current` subdirectory:

    ```bash
    cd alphafold_current
    git clone https://github.com/SzymonNowakowski/alphafold.git
    ```

1. Download chemical properties file (i.e. execute manually lines 62-63 from the Dockerfile) into the `alphafold/alphafold/common` subdirectory of `alphafold_current` (which has been set as the current directory in the previous step):
    ```
    wget -q -P ./alphafold/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
    ```

1. Submit the following slurm job `run_alphafold_external_code.slurm` with the `FASTA` type input file located in `inputs` directory passed as a parameter:

    ```bash
    #!/bin/bash
    #SBATCH --job-name alphafold2.1.1_external_code
    #SBATCH -A xxxx             # your account or grant name (to be charged for the computations in HPC)
    #SBATCH --time=2-00:00:00   # or whatever fits the QoS, adjust this to match the walltime of your job
    #SBATCH --cpus-per-task=8   # DO NOT INCREASE THIS AS ALPHAFOLD CANNOT TAKE ADVANTAGE OF MORE
    #SBATCH --gres=gpu:1        # You need to request one GPU to be able to run AlphaFold properly
    #SBATCH --mem=90G           # adjust this according to the memory requirement per node you need
    
    ###LOGGING
    echo $1 >> outputs/$SLURM_JOB_ID.desc
    cat $0 >> outputs/$SLURM_JOB_ID.desc
    
    #set the environment PATH
    export PYTHONNOUSERSITE=True
    #module load singularity
    ALPHAFOLD_DATA_PATH=remote_dir_with_protein_databases
    ALPHAFOLD_MODELS=remote_dir_with_model_parameters
    BASE_DIR=$(pwd)
    CODE_DIR=$BASE_DIR/alphafold_current/alphafold
    
    
    #Run the command
    singularity  exec --nv \
     -B $CODE_DIR:/alphafold_current \
     -B $ALPHAFOLD_DATA_PATH:/data \
     -B $ALPHAFOLD_MODELS \
     --pwd  /app/alphafold remote_dir/alphafold-2.1.1.sif \
      python $CODE_DIR/run_alphafold_external_code.py \
     --fasta_paths=$BASE_DIR/$1 \
     --uniref90_database_path=/data/uniref90/uniref90.fasta \
     --data_dir=/data \
     --mgnify_database_path=/data/mgnify/mgy_clusters_2018_12.fa \
     --bfd_database_path=/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
     --uniclust30_database_path=/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
     --pdb70_database_path=/data/pdb70/pdb70 \
     --template_mmcif_dir=/data/pdb_mmcif/mmcif_files \
     --obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat \
     --output_dir=$BASE_DIR/outputs \
     --model_preset=monomer \
     --db_preset=full_dbs \
     --max_template_date=2021-12-31 
    ```
    
    In particular, you may submit it with a command
    
    ```bash
    sbatch run_alphafold_external_code.slurm inputs/file_with_monomer.fasta
    ```
    
    The results would be then written into the `outputs/file_with_monomer` directory.

#### Execution divided into CPU- and GPU-intensive parts

Features extraction step is often the most time consuming part of the computations. You may separate feature extraction (CPU-intensive) and structure prediction (GPU-intensive) tasks as below:

1. **Feature extraction** step

   1. Increase `n_cpu` (default is `8`) to take advantage 
      of more cores available. However, many processes in 
      the preliminary Alphafold2 pipeline are not parallelized 
      and the overall computation time scales 
      poorly with number of cores.

   2. Submit the following slurm job `run_alphafold_features.slurm` with the `FASTA` type input file located in `inputs` directory passed as a parameter:
      ```bash
      #!/bin/bash
      #SBATCH --job-name alphafold2.1.1_features
      #SBATCH -A xxxx             # your account or grant name (to be charged for the computations in HPC)
      #SBATCH --time=2-00:00:00   # or whatever fits the QoS, adjust this to match the walltime of your job
      #SBATCH --cpus-per-task=8   # default 8. Shouldn't be larger than n_cpu parameter
      #SBATCH --gres=gpu:0        # You don't need GPU to feature extraction
      #SBATCH --mem=90G           # adjust this according to the memory requirement per node you need
   
      ###LOGGING
      echo $1 >> outputs/$SLURM_JOB_ID.desc
      cat $0 >> outputs/$SLURM_JOB_ID.desc
   
      #set the environment PATH
      export PYTHONNOUSERSITE=True
      #module load singularity
      ALPHAFOLD_DATA_PATH=remote_dir_with_protein_databases
      BASE_DIR=$(pwd)
   
      #Run the command
      singularity  exec \
       -B $ALPHAFOLD_DATA_PATH:/data \
       --pwd  /app/alphafold remote_dir/alphafold-2.1.1.sif \
       python run_alphafold_extract_features.py \
       --fasta_paths=$BASE_DIR/$1 \
       --uniref90_database_path=/data/uniref90/uniref90.fasta \
       --mgnify_database_path=/data/mgnify/mgy_clusters_2018_12.fa \
       --bfd_database_path=/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
       --uniclust30_database_path=/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
       --pdb70_database_path=/data/pdb70/pdb70 \
       --template_mmcif_dir=/data/pdb_mmcif/mmcif_files \
       --obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat \
       --output_dir=$BASE_DIR/outputs \
       --model_preset=monomer \
       --db_preset=full_dbs \
       --max_template_date=2021-12-31 \
       --max_template_hits=20 \
       --n_cpu=8
      ```

      In particular, you may submit it with a command

      ```bash
      sbatch run_alphafold_features.slurm inputs/file_with_monomer.fasta
      ```
   
      The feature file `features.pkl` would be then written into  `outputs/file_with_monomer` subdirectory.

2. **Predict structure from precomputed features** step

   :ledger: **Note 1: The `run_alphafold_from_features.py` script
   supports both the regular (running docker built-in codebase) and the external codebase way of running Alphafold2.**
   
   :ledger: **Note 2: The `run_alphafold_from_features.py` script
   accepts the `--structural_hypothesis_file` flag as optional. In the execution guidelines below it is provided, 
   but it may be ommited as well.**    
   1. If you wish to run the external codebase, execute steps 1-3 
      from Section [External codebase](#external-codebase) 
      *before* submitting the slurm job `run_alphafold_predict.slurm`. 
    
   2. If you wish to run the container codebase, execute the following two bash commands:
      ```bash
      mkdir alphafold_current
      mkdir alphafold_current/alphafold
       ``` 
      to create 
      a dummy empty codebase subdirectory structure (just to ensure the container binding doesn't fail)
      and then you just need to submit the slurm job `run_alphafold_predict.slurm` as explained below.  

   3. Submit the following slurm job `run_alphafold_predict.slurm` 
      with a path to `features.pkl` file (created in a previous *feature extraction* step in the subdirectory of the `outputs` directory) 
      and a path to an auxiliary hypothesis PDB file input parameters:
      ```bash
      #!/bin/bash
      #SBATCH --job-name alphafold2.1.1_predict
      #SBATCH -A xxxx             # your account or grant name (to be charged for the computations in HPC)
      #SBATCH --time=2-00:00:00   # or whatever fits the QoS, adjust this to match the walltime of your job
      #SBATCH --cpus-per-task=8   # DO NOT INCREASE THIS AS ALPHAFOLD CANNOT TAKE ADVANTAGE OF MORE
      #SBATCH --gres=gpu:1        # You need to request one GPU to be able to run AlphaFold properly
      #SBATCH --mem=90G           # adjust this according to the memory requirement per node you need
    
      ###LOGGING
      
      echo $1 >> outputs/$SLURM_JOB_ID.desc
      echo $2 >> outputs/$SLURM_JOB_ID.desc
      cat $0 >> outputs/$SLURM_JOB_ID.desc
   
      #set the environment PATH
      export PYTHONNOUSERSITE=True
      #module load singularity
      ALPHAFOLD_MODELS=remote_dir_with_model_parameters
      BASE_DIR=$(pwd)
      CODE_DIR=$BASE_DIR/alphafold_current/alphafold
   
      #Run the command
      singularity  exec --nv \
       -B $CODE_DIR:/alphafold_current \
       -B $ALPHAFOLD_MODELS:/data \
       --pwd  /app/alphafold remote_dir/alphafold-2.1.1.sif \
       python run_alphafold_from_features.py \
       --features_paths=$BASE_DIR/$1 \
       --data_dir=/data \
       --output_dir=$BASE_DIR/outputs \
       --model_preset=monomer \
       --structural_hypothesis_file=$BASE_DIR/$2    #Auxiliary structural hypothesis PDB file to warm-start alphafold iterations
      ```

      In particular, you may submit it with a command

      ```bash
      sbatch run_alphafold_from_features.slurm outputs/file_with_monomer/features.pkl inputs/file_with_monomer_structural_hypothesis.PDB
      ```
   
      The results of inference would be then written into the `outputs/file_with_monomer` subdirectory.

#### Changes
New scripts in this version (Center4ML version):

```bash
run_alphafold_external_code.py     #external codebase
run_alphafold_extract_features.py  #CPU-intensive part of computations only
run_alphafold_from_features.py     #GPU-intensive part of computations only, #
                                   # both container codebase and external codebase supported
                                   # also, supports structural_hypothesis_file parameter     
```

New flags in v2.1.1:

```bash
 --model_preset=monomer    
 --db_preset=full_dbs
``` 
New optional flag in v2.1.1 - **Center4ML version only**:

```bash
 --structural_hypothesis_file=path_to_auxiliary_PDB_file    $Auxiliary structural hypothesis PDB file to warm-start alphafold iterations
``` 
Also, it became mandatory to set `max_template_date` in v2.1.1

Flags from v2.0.0 no longer in use in v2.1.1:
```
 #--model_names='model_1' 
 #--preset=full_dbs
```


### Running AlphaFold-Multimer

All steps are the same as when running the monomer system, but you will have to

*   provide an input fasta with multiple sequences,
*   set `--model_preset=multimer`,
*   optionally set the `--is_prokaryote_list` flag with booleans that determine
    whether all input sequences in the given fasta file are prokaryotic. If that
    is not the case or the origin is unknown, set to `false` for that fasta.

An example that folds a protein complex `multimer.fasta` that is prokaryotic:

```bash
python3 docker/run_docker.py \
  --fasta_paths=multimer.fasta \
  --is_prokaryote_list=true \
  --max_template_date=2020-05-14 \
  --model_preset=multimer \
  --data_dir=$DOWNLOAD_DIR
```

### Examples

Below are examples on how to use AlphaFold in different scenarios.

#### Folding a monomer

Say we have a monomer with the sequence `<SEQUENCE>`. The input fasta should be:

```fasta
>sequence_name
<SEQUENCE>
```

Then run the following command:

```bash
python3 docker/run_docker.py \
  --fasta_paths=monomer.fasta \
  --max_template_date=2021-11-01 \
  --model_preset=monomer \
  --data_dir=$DOWNLOAD_DIR
```

#### Folding a homomer

Say we have a homomer from a prokaryote with 3 copies of the same sequence
`<SEQUENCE>`. The input fasta should be:

```fasta
>sequence_1
<SEQUENCE>
>sequence_2
<SEQUENCE>
>sequence_3
<SEQUENCE>
```

Then run the following command:

```bash
python3 docker/run_docker.py \
  --fasta_paths=homomer.fasta \
  --is_prokaryote_list=true \
  --max_template_date=2021-11-01 \
  --model_preset=multimer \
  --data_dir=$DOWNLOAD_DIR
```

#### Folding a heteromer

Say we have a heteromer A2B3 of unknown origin, i.e. with 2 copies of
`<SEQUENCE A>` and 3 copies of `<SEQUENCE B>`. The input fasta should be:

```fasta
>sequence_1
<SEQUENCE A>
>sequence_2
<SEQUENCE A>
>sequence_3
<SEQUENCE B>
>sequence_4
<SEQUENCE B>
>sequence_5
<SEQUENCE B>
```

Then run the following command:

```bash
python3 docker/run_docker.py \
  --fasta_paths=heteromer.fasta \
  --is_prokaryote_list=false \
  --max_template_date=2021-11-01 \
  --model_preset=multimer \
  --data_dir=$DOWNLOAD_DIR
```

#### Folding multiple monomers one after another

Say we have a two monomers, `monomer1.fasta` and `monomer2.fasta`.

We can fold both sequentially by using the following command:

```bash
python3 docker/run_docker.py \
  --fasta_paths=monomer1.fasta,monomer2.fasta \
  --max_template_date=2021-11-01 \
  --model_preset=monomer \
  --data_dir=$DOWNLOAD_DIR
```

#### Folding multiple multimers one after another

Say we have a two multimers, `multimer1.fasta` and `multimer2.fasta`. Both are
from a prokaryotic organism.

We can fold both sequentially by using the following command:

```bash
python3 docker/run_docker.py \
  --fasta_paths=multimer1.fasta,multimer2.fasta \
  --is_prokaryote_list=true,true \
  --max_template_date=2021-11-01 \
  --model_preset=multimer \
  --data_dir=$DOWNLOAD_DIR
```

### AlphaFold output

The outputs will be saved in a subdirectory of the directory provided via the
`--output_dir` flag of `run_docker.py` (defaults to `/tmp/alphafold/`). The
outputs include the computed MSAs, unrelaxed structures, relaxed structures,
ranked structures, raw model outputs, prediction metadata, and section timings.
The `--output_dir` directory will have the following structure:

```
<target_name>/
    features.pkl
    ranked_{0,1,2,3,4}.pdb
    ranking_debug.json
    relaxed_model_{1,2,3,4,5}.pdb
    result_model_{1,2,3,4,5}.pkl
    timings.json
    unrelaxed_model_{1,2,3,4,5}.pdb
    msas/
        bfd_uniclust_hits.a3m
        mgnify_hits.sto
        uniref90_hits.sto
```

The contents of each output file are as follows:

*   `features.pkl` – A `pickle` file containing the input feature NumPy arrays
    used by the models to produce the structures.
*   `unrelaxed_model_*.pdb` – A PDB format text file containing the predicted
    structure, exactly as outputted by the model.
*   `relaxed_model_*.pdb` – A PDB format text file containing the predicted
    structure, after performing an Amber relaxation procedure on the unrelaxed
    structure prediction (see Jumper et al. 2021, Suppl. Methods 1.8.6 for
    details).
*   `ranked_*.pdb` – A PDB format text file containing the relaxed predicted
    structures, after reordering by model confidence. Here `ranked_0.pdb` should
    contain the prediction with the highest confidence, and `ranked_4.pdb` the
    prediction with the lowest confidence. To rank model confidence, we use
    predicted LDDT (pLDDT) scores (see Jumper et al. 2021, Suppl. Methods 1.9.6
    for details).
*   `ranking_debug.json` – A JSON format text file containing the pLDDT values
    used to perform the model ranking, and a mapping back to the original model
    names.
*   `timings.json` – A JSON format text file containing the times taken to run
    each section of the AlphaFold pipeline.
*   `msas/` - A directory containing the files describing the various genetic
    tool hits that were used to construct the input MSA.
*   `result_model_*.pkl` – A `pickle` file containing a nested dictionary of the
    various NumPy arrays directly produced by the model. In addition to the
    output of the structure module, this includes auxiliary outputs such as:

    *   Distograms (`distogram/logits` contains a NumPy array of shape [N_res,
        N_res, N_bins] and `distogram/bin_edges` contains the definition of the
        bins).
    *   Per-residue pLDDT scores (`plddt` contains a NumPy array of shape
        [N_res] with the range of possible values from `0` to `100`, where `100`
        means most confident). This can serve to identify sequence regions
        predicted with high confidence or as an overall per-target confidence
        score when averaged across residues.
    *   Present only if using pTM models: predicted TM-score (`ptm` field
        contains a scalar). As a predictor of a global superposition metric,
        this score is designed to also assess whether the model is confident in
        the overall domain packing.
    *   Present only if using pTM models: predicted pairwise aligned errors
        (`predicted_aligned_error` contains a NumPy array of shape [N_res,
        N_res] with the range of possible values from `0` to
        `max_predicted_aligned_error`, where `0` means most confident). This can
        serve for a visualisation of domain packing confidence within the
        structure.

The pLDDT confidence measure is stored in the B-factor field of the output PDB
files (although unlike a B-factor, higher pLDDT is better, so care must be taken
when using for tasks such as molecular replacement).

This code has been tested to match mean top-1 accuracy on a CASP14 test set with
pLDDT ranking over 5 model predictions (some CASP targets were run with earlier
versions of AlphaFold and some had manual interventions; see our forthcoming
publication for details). Some targets such as T1064 may also have high
individual run variance over random seeds.

## Inferencing many proteins

The provided inference script is optimized for predicting the structure of a
single protein, and it will compile the neural network to be specialized to
exactly the size of the sequence, MSA, and templates. For large proteins, the
compile time is a negligible fraction of the runtime, but it may become more
significant for small proteins or if the multi-sequence alignments are already
precomputed. In the bulk inference case, it may make sense to use our
`make_fixed_size` function to pad the inputs to a uniform size, thereby reducing
the number of compilations required.

We do not provide a bulk inference script, but it should be straightforward to
develop on top of the `RunModel.predict` method with a parallel system for
precomputing multi-sequence alignments. Alternatively, this script can be run
repeatedly with only moderate overhead.

## Note on CASP14 reproducibility

AlphaFold's output for a small number of proteins has high inter-run variance,
and may be affected by changes in the input data. The CASP14 target T1064 is a
notable example; the large number of SARS-CoV-2-related sequences recently
deposited changes its MSA significantly. This variability is somewhat mitigated
by the model selection process; running 5 models and taking the most confident.

To reproduce the results of our CASP14 system as closely as possible you must
use the same database versions we used in CASP. These may not match the default
versions downloaded by our scripts.

For genetics:

*   UniRef90:
    [v2020_01](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2020_01/uniref/)
*   MGnify:
    [v2018_12](http://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2018_12/)
*   Uniclust30: [v2018_08](http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/)
*   BFD: [only version available](https://bfd.mmseqs.com/)

For templates:

*   PDB: (downloaded 2020-05-14)
*   PDB70: [2020-05-13](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200513.tar.gz)

An alternative for templates is to use the latest PDB and PDB70, but pass the
flag `--max_template_date=2020-05-14`, which restricts templates only to
structures that were available at the start of CASP14.

## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{AlphaFold2021,
  author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A A and Ballard, Andrew J and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
  journal = {Nature},
  title   = {Highly accurate protein structure prediction with {AlphaFold}},
  year    = {2021},
  volume  = {596},
  number  = {7873},
  pages   = {583--589},
  doi     = {10.1038/s41586-021-03819-2}
}
```

In addition, if you use the AlphaFold-Multimer mode, please cite:

```bibtex
@article {AlphaFold-Multimer2021,
  author       = {Evans, Richard and O{\textquoteright}Neill, Michael and Pritzel, Alexander and Antropova, Natasha and Senior, Andrew and Green, Tim and {\v{Z}}{\'\i}dek, Augustin and Bates, Russ and Blackwell, Sam and Yim, Jason and Ronneberger, Olaf and Bodenstein, Sebastian and Zielinski, Michal and Bridgland, Alex and Potapenko, Anna and Cowie, Andrew and Tunyasuvunakool, Kathryn and Jain, Rishub and Clancy, Ellen and Kohli, Pushmeet and Jumper, John and Hassabis, Demis},
  journal      = {bioRxiv}
  title        = {Protein complex prediction with AlphaFold-Multimer},
  year         = {2021},
  elocation-id = {2021.10.04.463034},
  doi          = {10.1101/2021.10.04.463034},
  URL          = {https://www.biorxiv.org/content/early/2021/10/04/2021.10.04.463034},
  eprint       = {https://www.biorxiv.org/content/early/2021/10/04/2021.10.04.463034.full.pdf},
}
```

## Community contributions

Colab notebooks provided by the community (please note that these notebooks may
vary from our full AlphaFold system and we did not validate their accuracy):

*   The [ColabFold AlphaFold2 notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb)
    by Martin Steinegger, Sergey Ovchinnikov and Milot Mirdita, which uses an
    API hosted at the Södinglab based on the MMseqs2 server [(Mirdita et al.
    2019, Bioinformatics)](https://academic.oup.com/bioinformatics/article/35/16/2856/5280135)
    for the multiple sequence alignment creation.

## Acknowledgements

AlphaFold communicates with and/or references the following separate libraries
and packages:

*   [Abseil](https://github.com/abseil/abseil-py)
*   [Biopython](https://biopython.org)
*   [Chex](https://github.com/deepmind/chex)
*   [Colab](https://research.google.com/colaboratory/)
*   [Docker](https://www.docker.com)
*   [HH Suite](https://github.com/soedinglab/hh-suite)
*   [HMMER Suite](http://eddylab.org/software/hmmer)
*   [Haiku](https://github.com/deepmind/dm-haiku)
*   [Immutabledict](https://github.com/corenting/immutabledict)
*   [JAX](https://github.com/google/jax/)
*   [Kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi)
*   [matplotlib](https://matplotlib.org/)
*   [ML Collections](https://github.com/google/ml_collections)
*   [NumPy](https://numpy.org)
*   [OpenMM](https://github.com/openmm/openmm)
*   [OpenStructure](https://openstructure.org)
*   [pandas](https://pandas.pydata.org/)
*   [pymol3d](https://github.com/avirshup/py3dmol)
*   [SciPy](https://scipy.org)
*   [Sonnet](https://github.com/deepmind/sonnet)
*   [TensorFlow](https://github.com/tensorflow/tensorflow)
*   [Tree](https://github.com/deepmind/tree)
*   [tqdm](https://github.com/tqdm/tqdm)

We thank all their contributors and maintainers!

## License and Disclaimer

This is not an officially supported Google product.

Copyright 2021 DeepMind Technologies Limited.

### AlphaFold Code License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

### Model Parameters License

The AlphaFold parameters are made available for non-commercial use only, under
the terms of the Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0) license. You can find details at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode

### Third-party software

Use of the third-party software, libraries or code referred to in the
[Acknowledgements](#acknowledgements) section above may be governed by separate
terms and conditions or license provisions. Your use of the third-party
software, libraries or code is subject to any such terms and you should check
that you can comply with any applicable restrictions or terms and conditions
before use.

### Mirrored Databases

The following databases have been mirrored by DeepMind, and are available with reference to the following:

*   [BFD](https://bfd.mmseqs.com/) (unmodified), by Steinegger M. and Söding J., available under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

*   [BFD](https://bfd.mmseqs.com/) (modified), by Steinegger M. and Söding J., modified by DeepMind, available under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/). See the Methods section of the [AlphaFold proteome paper](https://www.nature.com/articles/s41586-021-03828-1) for details.

*   [Uniclust30: v2018_08](http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/) (unmodified), by Mirdita M. et al., available under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

*   [MGnify: v2018_12](http://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/current_release/README.txt) (unmodified), by Mitchell AL et al., available free of all copyright restrictions and made fully and freely available for both non-commercial and commercial use under [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
