#! /usr/bin/env python3

import asyncio

from graph.graph import Graph, Node


async def main() -> None:
    graph = Graph()
    node = Node(id="1")
    graph.add_node(node)
    graph.test()


if __name__ == "__main__":
    asyncio.run(main())
