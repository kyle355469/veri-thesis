# System Requirements

## Functionality
- Top‑level module that wires all sub‑blocks together, handles clock‑domain crossing, and implements the optional SD_CLK_SEP logic. It must expose a Wishbone slave interface for configuration, a Wishbone master interface for data transfer, SDIO signals (clk, cmd, dat[3:0]), and an IRQ output.

## Performance
- {'wishbone_clk': '≤ 100\u202fMHz', 'sd_clk': '≤ 48\u202fMHz (max)', 'cdc_fifo_depth': '≥ 64 words (configurable)', 'latency': '≤ 10\u202fcycles from Wishbone write to SDIO data output'}

## I/O Interfaces
- TBD

## Protocols
- TBD

## PPA Constraints
- TBD

## Clock / Reset
- TBD

## Assumptions
- TBD
