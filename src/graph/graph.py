#! /usr/bin/env python3

from pydantic import BaseModel


class Node(BaseModel):
    id: str


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def test(self) -> None:
        print("test")
