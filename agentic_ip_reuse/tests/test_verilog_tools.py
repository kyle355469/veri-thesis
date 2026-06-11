import tempfile
import unittest
from pathlib import Path

from agentic_ip_reuse.verilog_tools import (
    check_port_compatibility,
    generate_rtl_module,
    validate_verilog,
)

SIMPLE_MODULE = """\
module simple_fifo #(parameter DATA_WIDTH = 8) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [DATA_WIDTH-1:0] din,
    input  wire                  wr_en,
    output wire [DATA_WIDTH-1:0] dout,
    output wire                  empty
);
endmodule
"""

CONSUMER_MODULE = """\
module consumer (
    input  wire [7:0] data_in,
    input  wire       valid,
    output wire       ready
);
endmodule
"""

PRODUCER_MODULE = """\
module producer (
    output wire [7:0] data_out,
    output wire       valid,
    input  wire       ready
);
endmodule
"""

WIDTH_MISMATCH_MODULE = """\
module wide_consumer (
    input  wire [15:0] data_in,
    input  wire        valid
);
endmodule
"""

DIRECTION_MISMATCH_MODULE = """\
module bad_producer (
    output wire [7:0] data_out,
    output wire       valid,
    output wire       ready
);
endmodule
"""


class TestGenerateRtlModule(unittest.TestCase):
    def test_writes_sv_file_and_detects_ports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = generate_rtl_module(
                module_name="simple_fifo",
                file_path="rtl/simple_fifo.sv",
                verilog_code=SIMPLE_MODULE,
                description="Synchronous FIFO",
                output_dir=out,
            )
            self.assertTrue(result["ok"])
            self.assertTrue(Path(result["path"]).exists())
            self.assertEqual(result["module_name"], "simple_fifo")
            port_names = {p["name"] for p in result["ports_detected"]}
            self.assertIn("clk", port_names)
            self.assertIn("din", port_names)
            self.assertIn("dout", port_names)

    def test_strips_markdown_fence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fenced = "```verilog\n" + SIMPLE_MODULE + "\n```"
            result = generate_rtl_module("simple_fifo", "rtl/x.sv", fenced, "test", out)
            code = Path(result["path"]).read_text(encoding="utf-8")
            self.assertNotIn("```", code)
            self.assertIn("module simple_fifo", code)

    def test_port_direction_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_rtl_module("producer", "p.sv", PRODUCER_MODULE, "test", Path(tmp))
            ports = {p["name"]: p for p in result["ports_detected"]}
            self.assertEqual(ports["data_out"]["direction"], "output")
            self.assertEqual(ports["ready"]["direction"], "input")

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                generate_rtl_module("x", "../../etc/passwd", "module x; endmodule", "test", Path(tmp))


class TestValidateVerilog(unittest.TestCase):
    def test_valid_file_returns_no_errors_in_offline_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_rtl_module("simple_fifo", "rtl/fifo.sv", SIMPLE_MODULE, "test", out)
            result = validate_verilog("rtl/fifo.sv", out)
            self.assertIn("linter", result)
            self.assertIsInstance(result["errors"], list)

    def test_module_endmodule_mismatch_caught_offline(self):
        broken = "module broken_mod (\n  input wire clk\n);\n// missing endmodule"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_rtl_module("broken_mod", "rtl/broken.sv", broken, "test", out)
            result = validate_verilog("rtl/broken.sv", out)
            if result["linter"] == "offline_regex":
                self.assertFalse(result["ok"])
                self.assertTrue(any("endmodule" in e for e in result["errors"]))

    def test_missing_file_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_verilog("nonexistent.sv", Path(tmp))
            self.assertFalse(result["ok"])
            self.assertIn("error", result)


class TestCheckPortCompatibility(unittest.TestCase):
    def test_compatible_producer_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_rtl_module("producer", "producer.sv", PRODUCER_MODULE, "test", out)
            generate_rtl_module("consumer", "consumer.sv", CONSUMER_MODULE, "test", out)
            result = check_port_compatibility(
                "producer.sv", "consumer.sv",
                [{"a": "data_out", "b": "data_in"}, {"a": "valid", "b": "valid"}],
                out,
            )
            self.assertTrue(result["ok"])
            self.assertEqual(len(result["issues"]), 0)
            self.assertEqual(len(result["matched"]), 2)

    def test_width_mismatch_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_rtl_module("producer", "producer.sv", PRODUCER_MODULE, "test", out)
            generate_rtl_module("wide_consumer", "wide.sv", WIDTH_MISMATCH_MODULE, "test", out)
            result = check_port_compatibility(
                "producer.sv", "wide.sv",
                [{"a": "data_out", "b": "data_in"}],
                out,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(any("width" in issue["issue"] for issue in result["issues"]))

    def test_direction_mismatch_reported(self):
        # Connecting two output ports on the same net = two drivers = mismatch
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_rtl_module("producer", "producer.sv", PRODUCER_MODULE, "test", out)
            generate_rtl_module("bad", "bad.sv", DIRECTION_MISMATCH_MODULE, "test", out)
            # producer.valid is output, bad_producer.valid is also output → two drivers
            result = check_port_compatibility(
                "producer.sv", "bad.sv",
                [{"a": "valid", "b": "valid"}],
                out,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(any("direction" in issue["issue"] for issue in result["issues"]))

    def test_empty_port_pairs_returns_all_ports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_rtl_module("producer", "producer.sv", PRODUCER_MODULE, "test", out)
            generate_rtl_module("consumer", "consumer.sv", CONSUMER_MODULE, "test", out)
            result = check_port_compatibility("producer.sv", "consumer.sv", [], out)
            self.assertTrue(result["ok"])
            self.assertIn("ports_a", result)
            self.assertIn("ports_b", result)

    def test_missing_port_name_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_rtl_module("producer", "producer.sv", PRODUCER_MODULE, "test", out)
            generate_rtl_module("consumer", "consumer.sv", CONSUMER_MODULE, "test", out)
            result = check_port_compatibility(
                "producer.sv", "consumer.sv",
                [{"a": "nonexistent_port", "b": "data_in"}],
                out,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(any("not found" in issue["issue"] for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
