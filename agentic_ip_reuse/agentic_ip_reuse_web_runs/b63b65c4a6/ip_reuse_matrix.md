# IP Reuse Matrix

| Module | Selected IP | New RTL Required | Required Adapters | Risk Notes |
| --- | --- | --- | --- | --- |
| Buffer / FIFO | sync_fifo | False | Map data valid-ready to FIFO push/pop if native ports differ | Confirm almost-full/almost-empty semantics |
| Processing Core | None | True | None | No computation-specific IP selected in mock run |
