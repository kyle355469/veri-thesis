# Integration Plan

- {'pair': ['sdc_controller', 'wb_slave_if'], 'connection': {'sdc_controller.wb_slave_*': 'wb_slave_if.*'}, 'compatibility_check': 'passed'}
- {'pair': ['sdc_controller', 'wb_master_if'], 'connection': {'sdc_controller.wb_master_*': 'wb_master_if.*'}, 'compatibility_check': 'passed'}
- {'pair': ['sdc_controller', 'cdc_fifo'], 'connection': {'sdc_controller.core_to_fifo_*': 'cdc_fifo.*', 'sdc_controller.fifo_to_bus_*': 'cdc_fifo.*'}, 'compatibility_check': 'passed'}
- {'pair': ['sdc_controller', 'clk_sync'], 'connection': {'sdc_controller.core_rst_sync': 'clk_sync.in', 'sdc_controller.core_rst_sync_out': 'clk_sync.out'}, 'compatibility_check': 'passed'}
- {'pair': ['sdc_controller', 'sd_clk_sep_logic'], 'connection': {'sdc_controller.sd_clk_en': 'sd_clk_sep_logic.en', 'sdc_controller.sd_clk_o': 'sd_clk_sep_logic.out'}, 'compatibility_check': 'passed'}
- {'pair': ['sdc_controller', 'irq_enable_logic'], 'connection': {'sdc_controller.irq_en_i': 'irq_enable_logic.en', 'sdc_controller.irq_o': 'irq_enable_logic.out'}, 'compatibility_check': 'passed'}
- {'pair': ['sdc_controller', 'sdc_core_wrapper'], 'connection': {'sdc_controller.core_*': 'sdc_core_wrapper.*'}, 'compatibility_check': 'passed'}

## Unresolved Assumptions
- Exact timing of the SDC Core's data ready signal relative to its clock domain.
- Maximum burst length supported by the Wishbone master interface.
- Whether the SD_CLK_SEP must be phase‑aligned with the core clock.
- The width of the interrupt vector (assumed 1‑bit).
