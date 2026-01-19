# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform

from usf_bios.system_guard import guard_cli_entry
guard_cli_entry()

from usf_bios.pipelines import eval_main

if __name__ == '__main__':
    eval_main()
