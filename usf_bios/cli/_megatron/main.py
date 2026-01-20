# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
# Minimal Megatron CLI entry point - all logic is in compiled _core module

from usf_bios.cli._core import cli_main_megatron as cli_main, MEGATRON_ROUTE_MAPPING as ROUTE_MAPPING

if __name__ == '__main__':
    cli_main()
