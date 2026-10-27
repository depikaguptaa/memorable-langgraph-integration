# Module summary

## queue/core.py
- push: Append an item to the queue.
- pop: Remove and return the head item.

## queue/dlq.py
- park: Move a failed item to the dead-letter queue.

## queue/worker.py
- work: Drain the queue applying fn to each item.
