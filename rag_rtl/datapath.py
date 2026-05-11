from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .embeddings import Embedder
from .types import RtlDocument
from .vector_store import build_vector_store

YOSYS_PATH_IN_NAME_RE = re.compile(r"/(?:[^/:$]+/)*([^/:$]+:\d+)")


INPUT_PORTS_BY_CELL = {
    "$add": {"A", "B"},
    "$and": {"A", "B"},
    "$div": {"A", "B"},
    "$eq": {"A", "B"},
    "$ge": {"A", "B"},
    "$gt": {"A", "B"},
    "$le": {"A", "B"},
    "$logic_and": {"A", "B"},
    "$logic_not": {"A"},
    "$logic_or": {"A", "B"},
    "$lt": {"A", "B"},
    "$mod": {"A", "B"},
    "$mul": {"A", "B"},
    "$mux": {"A", "B", "S"},
    "$ne": {"A", "B"},
    "$not": {"A"},
    "$or": {"A", "B"},
    "$pmux": {"A", "B", "S"},
    "$reduce_and": {"A"},
    "$reduce_bool": {"A"},
    "$reduce_or": {"A"},
    "$reduce_xor": {"A"},
    "$shl": {"A", "B"},
    "$shr": {"A", "B"},
    "$sshl": {"A", "B"},
    "$sshr": {"A", "B"},
    "$sub": {"A", "B"},
    "$xor": {"A", "B"},
}

OUTPUT_PORTS_BY_CELL = {
    "$add": {"Y"},
    "$and": {"Y"},
    "$adff": {"Q"},
    "$adffe": {"Q"},
    "$dff": {"Q"},
    "$dffe": {"Q"},
    "$div": {"Y"},
    "$eq": {"Y"},
    "$ge": {"Y"},
    "$gt": {"Y"},
    "$le": {"Y"},
    "$logic_and": {"Y"},
    "$logic_not": {"Y"},
    "$logic_or": {"Y"},
    "$lt": {"Y"},
    "$mod": {"Y"},
    "$mul": {"Y"},
    "$mux": {"Y"},
    "$ne": {"Y"},
    "$not": {"Y"},
    "$or": {"Y"},
    "$pmux": {"Y"},
    "$reduce_and": {"Y"},
    "$reduce_bool": {"Y"},
    "$reduce_or": {"Y"},
    "$reduce_xor": {"Y"},
    "$shl": {"Y"},
    "$shr": {"Y"},
    "$sshl": {"Y"},
    "$sshr": {"Y"},
    "$sub": {"Y"},
    "$xor": {"Y"},
}


@dataclass
class DatapathNode:
    node_id: str
    kind: str
    label: str
    width: int = 1
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatapathEdge:
    source: str
    target: str
    kind: str
    width: int = 1
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatapathGraph:
    graph_id: str
    source_doc_id: str
    module: str
    nodes: List[DatapathNode]
    edges: List[DatapathEdge]
    operations: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def retrieval_text(self, max_edges: int = 80) -> str:
        ports = [
            node
            for node in self.nodes
            if node.kind == "port"
        ]
        op_text = " ".join(f"{op}:{count}" for op, count in sorted(self.operations.items()))
        port_text = " ".join(
            f"{node.attrs.get('direction', 'port')} {node.label}[{node.width}]"
            for node in sorted(ports, key=lambda item: item.label)
        )
        dependency_lines = []
        for edge in self.edges:
            if edge.kind != "dependency":
                continue
            via = edge.attrs.get("cell_type", "")
            src = edge.attrs.get("source_signal", edge.source)
            dst = edge.attrs.get("target_signal", edge.target)
            dependency_lines.append(f"{src} -> {dst} via {via}".strip())
            if len(dependency_lines) >= max_edges:
                break
        return "\n".join(
            part
            for part in [
                f"datapath graph {self.graph_id}",
                f"module {self.module}",
                f"ports {port_text}",
                f"operations {op_text}",
                "dependencies",
                "\n".join(dependency_lines),
            ]
            if part
        ).strip()


@dataclass
class DatapathIndexStats:
    source_documents: int = 0
    graphs: int = 0
    skipped: int = 0
    failures_path: Optional[str] = None


@dataclass
class _DatapathExtractionResult:
    document: RtlDocument
    graphs: List["DatapathGraph"] = field(default_factory=list)
    error: Optional[str] = None


class YosysDatapathExtractor:
    def __init__(self, yosys_bin: str = "yosys", timeout_s: int = 30):
        self.yosys_bin = yosys_bin
        self.timeout_s = timeout_s

    def extract_document(self, document: RtlDocument) -> List[DatapathGraph]:
        yosys_json = self._run_yosys_json(document.solution)
        return datapath_graphs_from_yosys_json(
            yosys_json,
            source_doc_id=document.doc_id,
            source_metadata=document.metadata,
        )

    def _run_yosys_json(self, rtl: str) -> Dict[str, Any]:
        if shutil.which(self.yosys_bin) is None:
            raise RuntimeError(f"{self.yosys_bin} not found on PATH")
        with tempfile.TemporaryDirectory(prefix="rag_rtl_datapath_") as tempdir:
            rtl_path = Path(tempdir) / "source.v"
            json_path = Path(tempdir) / "datapath.json"
            rtl_path.write_text(rtl, encoding="utf-8")
            script = (
                f"read_verilog -sv {rtl_path}; "
                "hierarchy -check; "
                "proc; opt; memory; opt; "
                f"write_json {json_path}"
            )
            completed = subprocess.run(
                [self.yosys_bin, "-q", "-p", script],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            if completed.returncode != 0:
                message = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(message or f"{self.yosys_bin} failed with {completed.returncode}")
            return json.loads(json_path.read_text(encoding="utf-8"))


def datapath_graphs_from_yosys_json(
    yosys_json: Mapping[str, Any],
    source_doc_id: str,
    source_metadata: Optional[Mapping[str, Any]] = None,
) -> List[DatapathGraph]:
    modules = yosys_json.get("modules", {})
    graphs: List[DatapathGraph] = []
    if not isinstance(modules, Mapping):
        return graphs
    for module_name, module_payload in sorted(modules.items()):
        if not isinstance(module_payload, Mapping):
            continue
        graphs.append(_build_module_graph(source_doc_id, module_name, module_payload, source_metadata or {}))
    return graphs


def graph_documents_from_datapaths(
    datapaths: Iterable[Tuple[RtlDocument, DatapathGraph]],
) -> List[RtlDocument]:
    graph_documents: List[RtlDocument] = []
    for source_document, graph in datapaths:
        graph_documents.append(
            RtlDocument(
                doc_id=graph.graph_id,
                problem=source_document.problem,
                solution=graph.retrieval_text(),
                tags=sorted(set(source_document.tags + ["datapath", "graph"] + list(graph.operations.keys()))),
                metadata={
                    "source_doc_id": source_document.doc_id,
                    "module": graph.module,
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                    "operations": graph.operations,
                    "source": "datapath_graph",
                },
            )
        )
    return graph_documents


def build_datapath_vector_db(
    documents: Iterable[RtlDocument],
    embedder: Embedder,
    output: str | Path,
    yosys_bin: str = "yosys",
    timeout_s: int = 30,
    jobs: int = 1,
) -> DatapathIndexStats:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datapaths_path = output / "datapaths.jsonl"
    failures_path = output / "failures.jsonl"
    jobs = max(1, int(jobs))
    stats = DatapathIndexStats(failures_path=str(failures_path))
    graph_pairs: List[Tuple[RtlDocument, DatapathGraph]] = []

    with datapaths_path.open("w", encoding="utf-8") as datapaths_handle, failures_path.open("w", encoding="utf-8") as failures_handle:
        for result in _iter_datapath_extractions(documents, yosys_bin=yosys_bin, timeout_s=timeout_s, jobs=jobs):
            document = result.document
            stats.source_documents += 1
            if result.error is not None:
                stats.skipped += 1
                failures_handle.write(
                    json.dumps(
                        {
                            "doc_id": document.doc_id,
                            "error": result.error,
                            "metadata": document.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            if not result.graphs:
                stats.skipped += 1
                failures_handle.write(
                    json.dumps({"doc_id": document.doc_id, "error": "no modules found"}, ensure_ascii=False) + "\n"
                )
                continue
            for graph in result.graphs:
                stats.graphs += 1
                graph_pairs.append((document, graph))
                datapaths_handle.write(json.dumps(graph.to_dict(), ensure_ascii=False) + "\n")

    graph_documents = graph_documents_from_datapaths(graph_pairs)
    if graph_documents:
        vectors = embedder.encode([document.retrieval_text for document in graph_documents])
    else:
        vectors = np.zeros((0, embedder.dim), dtype=np.float32)
    store = build_vector_store(graph_documents, np.asarray(vectors, dtype=np.float32))
    store.save(output)
    return stats


def _iter_datapath_extractions(
    documents: Iterable[RtlDocument],
    yosys_bin: str,
    timeout_s: int,
    jobs: int,
) -> Iterable[_DatapathExtractionResult]:
    if jobs <= 1:
        extractor = YosysDatapathExtractor(yosys_bin=yosys_bin, timeout_s=timeout_s)
        for document in documents:
            yield _extract_datapath_document(document, extractor)
        return

    def run(document: RtlDocument) -> _DatapathExtractionResult:
        extractor = YosysDatapathExtractor(yosys_bin=yosys_bin, timeout_s=timeout_s)
        return _extract_datapath_document(document, extractor)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        yield from executor.map(run, documents)


def _extract_datapath_document(
    document: RtlDocument,
    extractor: YosysDatapathExtractor,
) -> _DatapathExtractionResult:
    try:
        return _DatapathExtractionResult(document=document, graphs=extractor.extract_document(document))
    except Exception as exc:  # noqa: BLE001 - keep batch preprocessing resilient.
        return _DatapathExtractionResult(document=document, error=str(exc))


def _build_module_graph(
    source_doc_id: str,
    module_name: str,
    module_payload: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
) -> DatapathGraph:
    nodes: Dict[str, DatapathNode] = {}
    edges: List[DatapathEdge] = []
    bit_labels = _build_bit_labels(module_payload)

    def ensure_node(node: DatapathNode) -> None:
        nodes.setdefault(node.node_id, node)

    module_node_id = "module"
    ensure_node(DatapathNode(module_node_id, "module", module_name, attrs={"source_doc_id": source_doc_id}))

    ports = module_payload.get("ports", {})
    if isinstance(ports, Mapping):
        for port_name, port_payload in sorted(ports.items()):
            if not isinstance(port_payload, Mapping):
                continue
            bits = _coerce_bits(port_payload.get("bits", []))
            direction = str(port_payload.get("direction", "unknown"))
            port_id = f"port:{port_name}"
            ensure_node(
                DatapathNode(
                    port_id,
                    "port",
                    str(port_name),
                    width=max(len(bits), 1),
                    attrs={"direction": direction},
                )
            )
            edges.append(DatapathEdge(module_node_id, port_id, "has_port", width=max(len(bits), 1)))
            for bit in bits:
                net_id = _net_node_id(bit)
                ensure_node(DatapathNode(net_id, "net", bit_labels.get(_bit_key(bit), _format_bit(bit))))
                if direction == "input":
                    edges.append(DatapathEdge(port_id, net_id, "port_drives_net"))
                elif direction == "output":
                    edges.append(DatapathEdge(net_id, port_id, "net_drives_port"))
                else:
                    edges.append(DatapathEdge(port_id, net_id, "inout"))
                    edges.append(DatapathEdge(net_id, port_id, "inout"))

    cells = module_payload.get("cells", {})
    operation_counts: Counter[str] = Counter()
    if isinstance(cells, Mapping):
        for cell_index, (cell_name, cell_payload) in enumerate(sorted(cells.items())):
            if not isinstance(cell_payload, Mapping):
                continue
            clean_cell_name = _clean_yosys_name(str(cell_name))
            cell_type = str(cell_payload.get("type", "unknown"))
            operation_counts[cell_type] += 1
            cell_id = f"cell:{cell_index}:{clean_cell_name}"
            ensure_node(
                DatapathNode(
                    cell_id,
                    "cell",
                    clean_cell_name,
                    attrs={
                        "type": cell_type,
                        "parameters": cell_payload.get("parameters", {}),
                    },
                )
            )
            edges.append(DatapathEdge(module_node_id, cell_id, "has_cell", attrs={"cell_type": cell_type}))
            input_bits: List[Any] = []
            output_bits: List[Any] = []
            connections = cell_payload.get("connections", {})
            if not isinstance(connections, Mapping):
                continue
            for port_name, raw_bits in sorted(connections.items()):
                bits = _coerce_bits(raw_bits)
                direction = _cell_port_direction(cell_type, str(port_name), cell_payload)
                if direction == "output":
                    output_bits.extend(bits)
                elif direction == "input":
                    input_bits.extend(bits)
                for bit in bits:
                    net_id = _net_node_id(bit)
                    ensure_node(DatapathNode(net_id, "net", bit_labels.get(_bit_key(bit), _format_bit(bit))))
                    if direction == "output":
                        edges.append(
                            DatapathEdge(
                                cell_id,
                                net_id,
                                "cell_output",
                                attrs={"port": port_name, "cell_type": cell_type},
                            )
                        )
                    elif direction == "input":
                        edges.append(
                            DatapathEdge(
                                net_id,
                                cell_id,
                                "cell_input",
                                attrs={"port": port_name, "cell_type": cell_type},
                            )
                        )
                    else:
                        edges.append(
                            DatapathEdge(
                                net_id,
                                cell_id,
                                "cell_inout",
                                attrs={"port": port_name, "cell_type": cell_type},
                            )
                        )
                        edges.append(
                            DatapathEdge(
                                cell_id,
                                net_id,
                                "cell_inout",
                                attrs={"port": port_name, "cell_type": cell_type},
                            )
                        )
            for input_bit in input_bits:
                for output_bit in output_bits:
                    edges.append(
                        DatapathEdge(
                            _net_node_id(input_bit),
                            _net_node_id(output_bit),
                            "dependency",
                            attrs={
                                "cell": clean_cell_name,
                                "cell_type": cell_type,
                                "source_signal": bit_labels.get(_bit_key(input_bit), _format_bit(input_bit)),
                                "target_signal": bit_labels.get(_bit_key(output_bit), _format_bit(output_bit)),
                            },
                        )
                    )

    graph_id = f"{source_doc_id}:{module_name}"
    return DatapathGraph(
        graph_id=graph_id,
        source_doc_id=source_doc_id,
        module=module_name,
        nodes=list(nodes.values()),
        edges=edges,
        operations=dict(sorted(operation_counts.items())),
        metadata={
            "source": source_metadata.get("source"),
            "row": source_metadata.get("row"),
        },
    )


def _build_bit_labels(module_payload: Mapping[str, Any]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    netnames = module_payload.get("netnames", {})
    if not isinstance(netnames, Mapping):
        return labels
    for net_name, net_payload in sorted(netnames.items()):
        if not isinstance(net_payload, Mapping):
            continue
        bits = _coerce_bits(net_payload.get("bits", []))
        for index, bit in enumerate(bits):
            if len(bits) == 1:
                label = str(net_name)
            else:
                label = f"{net_name}[{index}]"
            labels.setdefault(_bit_key(bit), label)
    return labels


def _cell_port_direction(cell_type: str, port_name: str, cell_payload: Mapping[str, Any]) -> str:
    port_directions = cell_payload.get("port_directions", {})
    if isinstance(port_directions, Mapping):
        direction = port_directions.get(port_name)
        if direction in {"input", "output", "inout"}:
            return str(direction)
    if port_name in OUTPUT_PORTS_BY_CELL.get(cell_type, set()):
        return "output"
    if port_name in INPUT_PORTS_BY_CELL.get(cell_type, set()):
        return "input"
    if port_name.upper() in {"Y", "Q", "O", "OUT"}:
        return "output"
    return "input"


def _coerce_bits(raw_bits: Any) -> List[Any]:
    if isinstance(raw_bits, Sequence) and not isinstance(raw_bits, (str, bytes)):
        return list(raw_bits)
    if raw_bits is None:
        return []
    return [raw_bits]


def _net_node_id(bit: Any) -> str:
    return f"net:{_bit_key(bit)}"


def _bit_key(bit: Any) -> str:
    if isinstance(bit, int):
        return str(bit)
    return f"const:{bit}"


def _format_bit(bit: Any) -> str:
    if isinstance(bit, int):
        return f"bit{bit}"
    return str(bit)


def _clean_yosys_name(name: str) -> str:
    return YOSYS_PATH_IN_NAME_RE.sub(r"\1", name)
