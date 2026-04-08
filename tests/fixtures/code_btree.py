"""A B-tree implementation for ordered key-value storage.

B-trees are the backbone of most database indexes. Each node holds
multiple keys in sorted order, and internal nodes have one more child
than keys. This keeps the tree shallow even for millions of entries,
minimizing disk seeks.
"""

from __future__ import annotations
from typing import Generic, List, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class BTreeNode(Generic[K, V]):
    """A single node in the B-tree."""

    def __init__(self, leaf: bool = True):
        self.keys: List[K] = []
        self.values: List[V] = []
        self.children: List[BTreeNode[K, V]] = []
        self.leaf = leaf

    def is_full(self, order: int) -> bool:
        return len(self.keys) >= 2 * order - 1

    def __repr__(self) -> str:
        return f"BTreeNode(keys={self.keys}, leaf={self.leaf})"


class BTree(Generic[K, V]):
    """
    A B-tree of configurable order.

    Properties:
        - Every node has at most 2*order - 1 keys
        - Every internal node has at most 2*order children
        - Every non-root node has at least order - 1 keys
        - All leaves appear at the same depth
        - Search, insert, delete are O(log n)
    """

    def __init__(self, order: int = 3):
        if order < 2:
            raise ValueError("B-tree order must be at least 2")
        self.order = order
        self.root = BTreeNode[K, V](leaf=True)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def search(self, key: K) -> Optional[V]:
        return self._search(self.root, key)

    def insert(self, key: K, value: V) -> None:
        root = self.root

        if root.is_full(self.order):
            new_root = BTreeNode[K, V](leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root

        self._insert_non_full(self.root, key, value)
        self._size += 1

    def items(self) -> List[Tuple[K, V]]:
        result: List[Tuple[K, V]] = []
        self._inorder(self.root, result)
        return result

    def height(self) -> int:
        h = 0
        node = self.root
        while not node.leaf:
            h += 1
            node = node.children[0]
        return h

    def _search(self, node: BTreeNode[K, V], key: K) -> Optional[V]:
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i]

        if node.leaf:
            return None

        return self._search(node.children[i], key)

    def _insert_non_full(self, node: BTreeNode[K, V], key: K, value: V) -> None:
        i = len(node.keys) - 1

        if node.leaf:
            # Find insertion point and insert
            while i >= 0 and key < node.keys[i]:
                i -= 1
            node.keys.insert(i + 1, key)
            node.values.insert(i + 1, value)
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            if node.children[i].is_full(self.order):
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key, value)

    def _split_child(self, parent: BTreeNode[K, V], index: int) -> None:
        order = self.order
        child = parent.children[index]
        mid = order - 1

        new_node = BTreeNode[K, V](leaf=child.leaf)
        new_node.keys = child.keys[mid + 1:]
        new_node.values = child.values[mid + 1:]

        if not child.leaf:
            new_node.children = child.children[mid + 1:]
            child.children = child.children[:mid + 1]

        parent.keys.insert(index, child.keys[mid])
        parent.values.insert(index, child.values[mid])
        parent.children.insert(index + 1, new_node)

        child.keys = child.keys[:mid]
        child.values = child.values[:mid]

    def _inorder(self, node: BTreeNode[K, V], result: List[Tuple[K, V]]) -> None:
        for i in range(len(node.keys)):
            if not node.leaf:
                self._inorder(node.children[i], result)
            result.append((node.keys[i], node.values[i]))
        if not node.leaf:
            self._inorder(node.children[len(node.keys)], result)
