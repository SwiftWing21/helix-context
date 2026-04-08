"""A generic finite state machine with guard conditions and actions.

Used for modeling workflows, protocol handlers, and game logic.
Transitions can have guard conditions (predicates) and side-effect actions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar

S = TypeVar("S")  # State type
E = TypeVar("E")  # Event type


@dataclass
class Transition(Generic[S, E]):
    """A single state transition definition."""
    source: S
    event: E
    target: S
    guard: Optional[Callable[[Dict[str, Any]], bool]] = None
    action: Optional[Callable[[Dict[str, Any]], None]] = None


class StateMachine(Generic[S, E]):
    """
    A finite state machine with context-aware transitions.

    Features:
        - Guard conditions: transitions only fire if the guard returns True
        - Actions: side effects executed when a transition fires
        - Context: mutable dict passed to guards and actions
        - History: full transition log for debugging
        - Hooks: on_enter / on_exit callbacks per state
    """

    def __init__(self, initial_state: S, context: Optional[Dict[str, Any]] = None):
        self._state: S = initial_state
        self._context: Dict[str, Any] = context or {}
        self._transitions: List[Transition[S, E]] = []
        self._on_enter: Dict[S, List[Callable]] = {}
        self._on_exit: Dict[S, List[Callable]] = {}
        self._history: List[Dict[str, Any]] = []

    @property
    def state(self) -> S:
        return self._state

    @property
    def context(self) -> Dict[str, Any]:
        return self._context

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def add_transition(
        self,
        source: S,
        event: E,
        target: S,
        guard: Optional[Callable[[Dict[str, Any]], bool]] = None,
        action: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._transitions.append(Transition(source, event, target, guard, action))

    def on_enter(self, state: S, callback: Callable) -> None:
        self._on_enter.setdefault(state, []).append(callback)

    def on_exit(self, state: S, callback: Callable) -> None:
        self._on_exit.setdefault(state, []).append(callback)

    def send(self, event: E) -> bool:
        """
        Send an event to the state machine.
        Returns True if a transition fired, False if no matching transition.
        """
        for t in self._transitions:
            if t.source != self._state or t.event != event:
                continue

            if t.guard is not None and not t.guard(self._context):
                continue

            # Fire exit hooks
            for cb in self._on_exit.get(self._state, []):
                cb(self._context)

            # Execute action
            if t.action is not None:
                t.action(self._context)

            # Record history
            self._history.append({
                "from": self._state,
                "event": event,
                "to": t.target,
            })

            # Transition
            self._state = t.target

            # Fire enter hooks
            for cb in self._on_enter.get(self._state, []):
                cb(self._context)

            return True

        return False

    def can_send(self, event: E) -> bool:
        """Check if an event would trigger a transition from the current state."""
        for t in self._transitions:
            if t.source != self._state or t.event != event:
                continue
            if t.guard is not None and not t.guard(self._context):
                continue
            return True
        return False

    def available_events(self) -> Set[E]:
        """Return all events that can fire from the current state."""
        return {
            t.event for t in self._transitions
            if t.source == self._state
            and (t.guard is None or t.guard(self._context))
        }


# ── Example: Order workflow ─────────────────────────────────────────

def create_order_machine() -> StateMachine[str, str]:
    """Factory for a typical e-commerce order state machine."""
    sm = StateMachine("draft")

    # draft -> submitted (requires items in cart)
    sm.add_transition("draft", "submit", "pending",
        guard=lambda ctx: ctx.get("item_count", 0) > 0,
        action=lambda ctx: ctx.update({"submitted_at": "now"}),
    )

    # pending -> paid
    sm.add_transition("pending", "pay", "paid",
        action=lambda ctx: ctx.update({"paid": True}),
    )

    # paid -> shipped
    sm.add_transition("paid", "ship", "shipped",
        action=lambda ctx: ctx.update({"tracking": "TRACK-001"}),
    )

    # shipped -> delivered
    sm.add_transition("shipped", "deliver", "delivered")

    # any active state -> cancelled (with guard: can't cancel after shipping)
    for state in ["draft", "pending", "paid"]:
        sm.add_transition(state, "cancel", "cancelled")

    return sm
