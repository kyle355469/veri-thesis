# System Requirements

## Functionality
- Provide a Wishbone slave interface for configuration registers and BD access.
- Provide a Wishbone master interface for DMA transfers to/from external memory.
- Implement SD/MMC command and data path with CRC, interrupt, and status handling.
- Support buffer descriptor (BD) management for Tx and Rx.
- Provide dual‑clock FIFOs for Tx and Rx data transfer.
- Provide a clock divider for SD card clock generation.
- Support optional SD_CLK_SEP and SDC_IRQ_ENABLE configurations.

## Performance
- TBD

## I/O Interfaces
- TBD

## Protocols
- TBD

## PPA Constraints
- {'area': '< 200\u202fkGE', 'power': '< 1\u202fW (dynamic)', 'timing': 'setup/hold < 200\u202fps, max clock 200\u202fMHz'}

## Clock / Reset
- TBD

## Assumptions
- TBD
