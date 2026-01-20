# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform

from usf_bios.system_guard import guard_cli_entry
guard_cli_entry()

if __name__ == '__main__':
    from usf_bios.cli._core import sft_entry
    sft_entry()
