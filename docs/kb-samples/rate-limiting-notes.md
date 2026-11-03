---
title: Rate Limiting Notes
date: 2026-09-14
---

# Rate Limiting Notes

These are study notes on how services protect themselves from too many requests. They were written for this sample corpus and are free to use.

## Why rate limiting exists

Rate limiting caps how many requests a client can make in a given time window. It protects a service from overload, keeps one noisy client from starving the others, and limits the damage from abusive traffic such as password guessing. Most public APIs apply a limit per API key, per user, or per IP address.

A good limiter also tells clients what happened. When a request is rejected, the service should return HTTP status 429 (Too Many Requests) and a `Retry-After` header so well-behaved clients know when to try again.

## Common algorithms

### Token bucket

A bucket holds up to a fixed number of tokens. Each request removes one token, and tokens are added back at a steady rate. If the bucket is empty, the request is rejected. The bucket size sets how large a burst is allowed, and the refill rate sets the long-run average.

Token bucket is popular because it allows short bursts while still enforcing an average rate. A client that has been idle can send a quick burst of requests, and then settles into the refill rate.

```python
import time

class TokenBucket:
    def __init__(self, capacity, refill_per_sec):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_per_sec = refill_per_sec
        self.last = time.monotonic()

    def allow(self):
        now = time.monotonic()
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

### Fixed window counter

Time is split into fixed windows, such as one minute each. A counter tracks requests in the current window and resets when the window ends. It is very simple and cheap, but it has a known weakness: a client can send a full quota at the end of one window and another full quota at the start of the next, briefly doubling the allowed rate.

### Sliding window log

The limiter stores a timestamp for every request and counts how many fall inside the last window. This is accurate and avoids the boundary problem, but it uses more memory because every timestamp is kept.

### Sliding window counter

This approach blends the current and previous fixed-window counts, weighted by how far into the current window we are. It approximates a sliding window using only two counters, which makes it a good balance between accuracy and memory use.

## Where to enforce limits

Limits can live in an API gateway, in a middleware layer inside the application, or in a shared store such as Redis so that several servers see the same counts. When a service runs on many machines, a purely local counter lets each machine allow the full quota, so the real limit becomes much higher than intended. A shared store fixes this at the cost of an extra network call per request.

## Questions to revisit

- How should limits differ between free and paid clients?
- What happens to legitimate traffic if the shared store goes down: fail open or fail closed?
- Is a per-IP limit fair for users behind a shared office network?
