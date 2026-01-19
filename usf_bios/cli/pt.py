# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform

from usf_bios.system_guard import guard_cli_entry
guard_cli_entry()

if __name__ == '__main__':
    from usf_bios.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from usf_bios.pipelines import pretrain_main
    pretrain_main()
