import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_rtl.datapath import (
    DatapathGraph,
    DatapathNode,
    build_datapath_vector_db,
    datapath_graphs_from_yosys_json,
    graph_documents_from_datapaths,
)
from rag_rtl.embeddings import HashingEmbedder
from rag_rtl.types import RtlDocument
from rag_rtl.vector_store import VectorStore


class FakeExtractor:
    def __init__(self, yosys_bin="yosys", timeout_s=30):
        self.yosys_bin = yosys_bin
        self.timeout_s = timeout_s

    def extract_document(self, document):
        return [
            DatapathGraph(
                graph_id=f"{document.doc_id}:and2",
                source_doc_id=document.doc_id,
                module="and2",
                nodes=[
                    DatapathNode("module", "module", "and2"),
                    DatapathNode("cell:$and$1", "cell", "$and$1", attrs={"type": "$and"}),
                ],
                edges=[],
                operations={"$and": 1},
            )
        ]


class DatapathTests(unittest.TestCase):
    def test_builds_dependency_graph_from_yosys_json(self):
        yosys_json = {
            "modules": {
                "and2": {
                    "ports": {
                        "a": {"direction": "input", "bits": [2]},
                        "b": {"direction": "input", "bits": [3]},
                        "y": {"direction": "output", "bits": [4]},
                    },
                    "netnames": {
                        "a": {"bits": [2]},
                        "b": {"bits": [3]},
                        "y": {"bits": [4]},
                    },
                    "cells": {
                        "$and$1": {
                            "type": "$and",
                            "port_directions": {"A": "input", "B": "input", "Y": "output"},
                            "connections": {"A": [2], "B": [3], "Y": [4]},
                        }
                    },
                }
            }
        }

        graphs = datapath_graphs_from_yosys_json(yosys_json, source_doc_id="doc-1")

        self.assertEqual(len(graphs), 1)
        graph = graphs[0]
        self.assertEqual(graph.module, "and2")
        self.assertEqual(graph.operations, {"$and": 1})
        dependencies = [edge for edge in graph.edges if edge.kind == "dependency"]
        self.assertEqual(len(dependencies), 2)
        self.assertIn("a -> y via $and", graph.retrieval_text())
        self.assertIn("b -> y via $and", graph.retrieval_text())

    def test_graph_documents_keep_source_problem_and_graph_text(self):
        source = RtlDocument("doc-1", "Design an and gate", "module and2; endmodule", tags=["combinational"])
        graph = DatapathGraph(
            graph_id="doc-1:and2",
            source_doc_id="doc-1",
            module="and2",
            nodes=[DatapathNode("module", "module", "and2")],
            edges=[],
            operations={"$and": 1},
        )

        graph_documents = graph_documents_from_datapaths([(source, graph)])

        self.assertEqual(graph_documents[0].doc_id, "doc-1:and2")
        self.assertEqual(graph_documents[0].problem, "Design an and gate")
        self.assertIn("datapath graph doc-1:and2", graph_documents[0].solution)
        self.assertIn("datapath", graph_documents[0].tags)
        self.assertEqual(graph_documents[0].metadata["source_doc_id"], "doc-1")

    def test_build_datapath_vector_db_writes_structured_and_vector_artifacts(self):
        source = RtlDocument("doc-1", "Design an and gate", "module and2; endmodule")
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir)
            with patch("rag_rtl.datapath.YosysDatapathExtractor", FakeExtractor):
                stats = build_datapath_vector_db([source], HashingEmbedder(dim=64), output)

            self.assertEqual(stats.source_documents, 1)
            self.assertEqual(stats.graphs, 1)
            self.assertEqual(stats.skipped, 0)
            self.assertTrue((output / "datapaths.jsonl").exists())
            self.assertTrue((output / "documents.jsonl").exists())
            self.assertTrue((output / "vectors.npy").exists())

            graph_payload = json.loads((output / "datapaths.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(graph_payload["module"], "and2")
            store = VectorStore.load(output)
            self.assertEqual(store.documents[0].doc_id, "doc-1:and2")


if __name__ == "__main__":
    unittest.main()
