"""
oos/network.py
Realistic communication network:
  - Line-of-sight check (Earth occlusion)
  - Distance-based propagation delay + processing delay
  - Partial delivery (only reachable receivers get the message)
"""

import numpy as np

C_LIGHT = 3e8          # m/s
R_EARTH = 6_371_000.0  # m


def is_line_of_sight(r1: np.ndarray, r2: np.ndarray) -> bool:
    """Return True if r1 and r2 can see each other (Earth does not occlude)."""
    d = r2 - r1
    denom = np.dot(d, d)
    if denom == 0:
        return True
    t = np.clip(-np.dot(r1, d) / denom, 0.0, 1.0)
    closest = r1 + t * d
    return float(np.linalg.norm(closest)) > R_EARTH


class Network:
    """
    Decentralised message bus.

    Usage
    -----
    Call `broadcast(msg, sender_r, current_time, fleet)` to enqueue a message.
    Call `deliver(current_time, fleet)` every simulation step to push due
    messages into each receiver's `inbox`.
    """

    def __init__(self, processing_delay: float = 0.1):
        """
        Parameters
        ----------
        processing_delay : float
            Fixed extra delay (seconds) added on top of light-travel time.
        """
        self.processing_delay = processing_delay
        self._queue: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def broadcast(
        self,
        msg: dict,
        sender_r: np.ndarray,
        current_time: float,
        fleet: list,
    ) -> None:
        """
        Enqueue `msg` for every reachable fleet member (excluding sender).

        Earth-occlusion is checked immediately so we do not waste queue
        slots for signals that can never arrive.  Delay is computed from
        the sender's current position; if the receiver moves between now
        and delivery the delay is still correct (propagation is one-way).
        """
        sender_id = msg.get("oos_id", -1)
        
        for receiver in fleet:
            if receiver.id == sender_id:
                continue

            # # ---- Earth occlusion ----
            # if not is_line_of_sight(sender_r, receiver.r):
            #     continue

            # ---- Distance-based delay ----
            distance = float(np.linalg.norm(sender_r - receiver.r))
            prop_delay = distance / C_LIGHT
            total_delay = prop_delay + self.processing_delay
            deliver_time = current_time + total_delay

            self._queue.append(
                {
                    "msg": {**msg},          # shallow copy keeps arrays safe
                    "deliver_time": deliver_time,
                    "receiver_id": receiver.id,
                }
            )

    def deliver(self, current_time: float, fleet: list) -> int:
        """
        Push all messages whose delivery time has elapsed into receiver inboxes.

        Returns the number of messages delivered this step.
        """
        remaining = []
        delivered = 0

        for item in self._queue:
            if item["deliver_time"] <= current_time:
                receiver = next(
                    (o for o in fleet if o.id == item["receiver_id"]), None
                )
                if receiver is not None:
                    receiver.inbox.append(item["msg"])
                    delivered += 1
            else:
                remaining.append(item)

        self._queue = remaining
        return delivered

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def queue_size(self) -> int:
        return len(self._queue)