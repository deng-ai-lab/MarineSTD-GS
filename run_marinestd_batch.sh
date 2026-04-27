#!/usr/bin/env bash

# Batch training-and-render script for MarineSTD-GS.
# - scene_list: relative scene paths under the repository-level data/ directory.
# - EXP_TAG: experiment label used to group outputs/ and render_results/ entries.
# - The script assumes the default monocular depth folder is depthAnythingV2_u16.

set -u -o pipefail

clear

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=3

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT_DIR="${PROJECT_ROOT}/outputs"
RENDER_ROOT="${PROJECT_ROOT}/render_results"

#
# Example scenes for the public release.
# Replace these with your own relative scene paths under the repository-level data/ directory as needed.
#
scene_list=(
    "SeaThru_NeRF/Curasao"
    "SeaThru/D3"
    "Simulated_Dataset/S2-A-Med"
    "Simulated_Dataset/S2-A-Low"
    "Simulated_Dataset/S2-A-High"
    "Simulated_Dataset/S2-B-Med"
    "Simulated_Dataset/S2-C-Med"
    # "mmSynthetic/M3_3"
    # "mmSynthetic/M4_3"
    # "mmSynthetic/D4_3"
)

EXP_TAG="watch50_sh0/3stage_100_l1_womul_nega10000_depth_adadtv_1e_2_encode_bg"

# Default configuration used by the public release.
MODEL_CONFIG_COMMAND=" --pipeline.model.enable_negative_perturbation_regularization True"

# Recommended real-world configuration.
# In our experiments, the following settings were more stable on real underwater datasets.
# The opacity reset / post-densification pruning settings below are tied to the modified
# gsplat behavior used in this repository.
# MODEL_CONFIG_COMMAND=" \
# --pipeline.model.enable_perturbation_relu True \
# --pipeline.model.enable_reg_loss True \
# --pipeline.model.reset_alpha_value 0.5 \
# --pipeline.model.cull_alpha_thresh_post 0.1 \
# --pipeline.model.continue_cull_post_densification True \
# --pipeline.model.cull_alpha_thresh 0.5 \
# --pipeline.model.reset_alpha_every 5 \
# "

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

find_latest_config() {
    local run_root="$1"
    local latest_dir=""

    if [ ! -d "$run_root" ]; then
        return 1
    fi

    latest_dir=$(ls -td "$run_root"/*/ 2>/dev/null | head -n 1 || true)
    if [ -z "$latest_dir" ]; then
        return 1
    fi

    echo "${latest_dir%/}/config.yml"
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

if [ "${#scene_list[@]}" -eq 0 ]; then
    echo "[WARN] scene_list is empty. Uncomment or add scene paths before running this script."
    exit 0
fi

echo "######################################################################################################################################################"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Log root: ${LOG_ROOT_DIR}"
echo "[INFO] Render root: ${RENDER_ROOT}"
echo "######################################################################################################################################################"

for scene_rel_path in "${scene_list[@]}"; do
    data_name="${scene_rel_path%/*}"
    scene_name="${scene_rel_path##*/}"
    DATA_DIR="${PROJECT_ROOT}/data/${data_name}/${scene_name}"
    EXP_NAME="${EXP_TAG}/${data_name}/${scene_name}"
    LOG_DIR="${LOG_ROOT_DIR}/${EXP_NAME}/marinestd-gs"
    RENDER_OUTPUT_DIR="${RENDER_ROOT}/${EXP_TAG}/${data_name}/${scene_name}/marinestd-gs"

    echo "######################################################################################################################################################"
    echo "[INFO] Scene: ${scene_rel_path}"
    echo "[INFO] Data directory: ${DATA_DIR}"
    echo "[INFO] Experiment name: ${EXP_NAME}"
    echo "[INFO] Render output directory: ${RENDER_OUTPUT_DIR}"

    mkdir -p "${RENDER_OUTPUT_DIR}"

    ns-train marinestd-gs --experiment-name "${EXP_NAME}" ${MODEL_CONFIG_COMMAND} --vis tensorboard \
    --data "${DATA_DIR}" \
    --pipeline.datamanager.dataparser.colmap-path sparse/0 \
    --pipeline.datamanager.dataparser.images-path images_wb \
    --pipeline.datamanager.dataparser.depths-path depthAnythingV2_u16

    latest_config="$(find_latest_config "${LOG_DIR}")" || true

    if [ -z "${latest_config}" ]; then
        echo "[ERROR] Failed to locate the latest config.yml under: ${LOG_DIR}"
        continue
    fi

    echo "[INFO] Latest config: ${latest_config}"

    # Dataset render on the default evaluation split: depth only with gray colormap.
    ns-marinestd-render dataset --load-config "${latest_config}" --rendered-output-names depth --output-path "${RENDER_OUTPUT_DIR}" --colormap-options.colormap gray

    # If `--pipeline.datamanager.dataparser.eval-mode all` is added during training,
    # the model uses all images for reconstruction, which removes the train/test split.

    # Dataset render examples on the default evaluation split.
    # ns-marinestd-render dataset --load-config "${latest_config}" --rendered-output-names rgb --output-path "${RENDER_OUTPUT_DIR}"
    # ns-marinestd-render dataset --load-config "${latest_config}" --rendered-output-names rgb_spatiotemporal_degraded --output-path "${RENDER_OUTPUT_DIR}"
    # ns-marinestd-render dataset --load-config "${latest_config}" --rendered-output-names rgb rgb_spatial_degraded rgb_spatiotemporal_degraded --output-path "${RENDER_OUTPUT_DIR}"
    ns-marinestd-render dataset --load-config "${latest_config}" --rendered-output-names rgb rgb_spatial_degraded rgb_spatiotemporal_degraded rgb_attenuation_map rgb_backscatter_map rgb_caustic_pattern --output-path "${RENDER_OUTPUT_DIR}"

    # Dataset render examples on the training split.
    # These outputs are also saved separately into different folders.
    # ns-marinestd-render dataset --load-config "${latest_config}" --split train --rendered-output-names rgb --output-path "${RENDER_OUTPUT_DIR}"
    ns-marinestd-render dataset --load-config "${latest_config}" --split train --rendered-output-names rgb rgb_spatial_degraded rgb_spatiotemporal_degraded --output-path "${RENDER_OUTPUT_DIR}"
done

echo "######################################################################################################################################################"
echo "[INFO] Batch run finished."
