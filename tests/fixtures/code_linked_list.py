"""A doubly-linked list with iteration and reversal."""

from __future__ import annotations
from typing import TypeVar, Generic, Iterator, Optional

T = TypeVar("T")


class Node(Generic[T]):
    __slots__ = ("value", "prev", "next")

    def __init__(self, value: T, prev: Optional[Node[T]] = None, next: Optional[Node[T]] = None):
        self.value = value
        self.prev = prev
        self.next = next

    def __repr__(self) -> str:
        return f"Node({self.value!r})"


class DoublyLinkedList(Generic[T]):
    """A doubly-linked list supporting O(1) append/prepend and O(n) search."""

    def __init__(self):
        self._head: Optional[Node[T]] = None
        self._tail: Optional[Node[T]] = None
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[T]:
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def __reversed__(self) -> Iterator[T]:
        current = self._tail
        while current is not None:
            yield current.value
            current = current.prev

    def __contains__(self, value: T) -> bool:
        return self._find_node(value) is not None

    def append(self, value: T) -> None:
        node = Node(value, prev=self._tail)
        if self._tail is not None:
            self._tail.next = node
        self._tail = node
        if self._head is None:
            self._head = node
        self._size += 1

    def prepend(self, value: T) -> None:
        node = Node(value, next=self._head)
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node
        self._size += 1

    def remove(self, value: T) -> bool:
        node = self._find_node(value)
        if node is None:
            return False
        self._unlink(node)
        return True

    def reverse(self) -> None:
        current = self._head
        while current is not None:
            current.prev, current.next = current.next, current.prev
            current = current.prev
        self._head, self._tail = self._tail, self._head

    def to_list(self) -> list[T]:
        return list(self)

    def _find_node(self, value: T) -> Optional[Node[T]]:
        current = self._head
        while current is not None:
            if current.value == value:
                return current
            current = current.next
        return None

    def _unlink(self, node: Node[T]) -> None:
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
        self._size -= 1
