# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
# Minimal CLI entry point - all logic is in compiled _core module

from usf_bios.cli._core import cli_main_standard as cli_main, ROUTE_MAPPING

if __name__ == '__main__':
    cli_main()
