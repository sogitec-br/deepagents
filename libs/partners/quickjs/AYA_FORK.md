# Temporary AyA cancellation fix

This downstream build is `langchain-quickjs==0.3.9+aya.1`. It is not an official
LangChain release and is not published to PyPI. It requires the paired downstream
build `quickjs-rs==0.2.6+aya.1`; the official 0.2.6 lacks its public bridge helper.

The base is upstream commit `92955e789ed280130f51d0d099be02cccf524510`.
Commit `7b1356b` drains active evaluations and PTC/subagent host calls before
closing their context. Closing prevents new dispatches, including queued calls,
and repeated cancellation cannot interrupt resource shutdown.

The wheel is built from this package without editing installed modules.
Its release records the complete source commit and artifact SHA-256.
Consumers must pin the artifact URL and hash; do not use a moving branch.

The generic upstream contribution excludes this downstream version and dependency
pin. Follow the upstream issue approval and assignment process before opening
an external PR. After both official releases contain the fix, adopt them and
remove the temporary downstream sources from consumer locks.
