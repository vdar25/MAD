# shellcheck shell=bash
# hosts: useocpm2m-*
# OCI MI300X cluster (amd-arad / amd-rccl partitions): RoCE, eight CX7 rail NICs
# plus the mlx5_1 management NIC per node. settings.sh documents each site setting
# and its default; anything already set in the environment wins.

: "${MODEL_ROOTS:=/mnt/m2m_nobackup/models_blog:/shared_inference/models_blog}"
: "${LOG_ROOT:=${LOG_PATH:-/shared_inference/${USER:-$(id -un)}/model_blog_logs}}"
: "${EXTRA_MOUNTS:=/shared_inference /mnt/m2m_nobackup}"
# Nodes are multi-homed; the 10.158.x addresses are the ones the nodes reach each other on.
: "${FABRIC_SUBNET:=10.158.}"

# The rail NICs only connect nodes on the same rail; mlx5_1 works across rails.
# USE_CX7_NICS=1 selects the rails, for KV transfer and for NCCL and MoRI.
if [[ "${USE_CX7_NICS:-0}" == "1" ]]; then
    _oci_nics=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
else
    _oci_nics=mlx5_1
fi
: "${KV_IB_DEVICES:=${_oci_nics}}"
: "${NCCL_IB_HCA:=${_oci_nics}}"
: "${MORI_RDMA_DEVICES:=${_oci_nics}}"
unset _oci_nics

# RoCE v2 over IPv4 is GID index 3 here; traffic class 106 is DSCP 26 (AF31).
: "${NCCL_IB_GID_INDEX:=3}"
: "${MORI_IB_GID_INDEX:=3}"
: "${NCCL_IB_TC:=106}"
: "${NCCL_IB_SL:=0}"

# NCCL over the 400G CX7 RoCE fabric. Only traffic between nodes uses it: a
# server that spans nodes, not a 1P1D run.
: "${NCCL_IB_DISABLE:=0}"
: "${NCCL_CROSS_NIC:=1}"
: "${NCCL_NET_GDR_LEVEL:=3}"
: "${NCCL_IB_QPS_PER_CONNECTION:=4}"
: "${NCCL_IB_SPLIT_DATA_ON_QPS:=1}"
: "${NCCL_BUFFSIZE:=8388608}"
: "${NCCL_IB_TIMEOUT:=22}"
: "${NCCL_IB_RETRY_CNT:=7}"
: "${NCCL_IB_PCI_RELAXED_ORDERING:=1}"
: "${NCCL_IB_ADAPTIVE_ROUTING:=1}"

export NCCL_IB_HCA MORI_RDMA_DEVICES NCCL_IB_GID_INDEX MORI_IB_GID_INDEX NCCL_IB_TC NCCL_IB_SL \
    NCCL_IB_DISABLE NCCL_CROSS_NIC NCCL_NET_GDR_LEVEL NCCL_IB_QPS_PER_CONNECTION \
    NCCL_IB_SPLIT_DATA_ON_QPS NCCL_BUFFSIZE NCCL_IB_TIMEOUT NCCL_IB_RETRY_CNT \
    NCCL_IB_PCI_RELAXED_ORDERING NCCL_IB_ADAPTIVE_ROUTING
