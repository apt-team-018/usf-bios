# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
import os

# CRITICAL: Validate system restrictions BEFORE any operation
from usf_bios.system_guard import guard_cli_entry
guard_cli_entry()

if __name__ == '__main__':
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from usf_bios.megatron import megatron_export_main
    megatron_export_main()
