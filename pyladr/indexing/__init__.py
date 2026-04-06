"""Term indexing: discrimination trees, path indexing, literal indexing.

Provides discrimination tree implementations for efficient term retrieval:
- DiscrimWild: imperfect filter (fast, may have false positives)
- DiscrimBind: perfect filter (slower, produces substitutions)
- Mindex: unified interface (matches C mindex.c)
- LiteralIndex: pos/neg pair of Mindexes (matches C lindex.c)
- FeatureIndex: feature vector prefiltering for subsumption
"""

from pyladr.indexing.discrimination_tree import (
    DiscrimBind,
    DiscrimWild,
    IndexType,
    Mindex,
)
from pyladr.indexing.feature_index import FeatureIndex, FeatureVector
from pyladr.indexing.literal_index import LiteralIndex

__all__ = [
    "DiscrimBind",
    "DiscrimWild",
    "FeatureIndex",
    "FeatureVector",
    "IndexType",
    "LiteralIndex",
    "Mindex",
]
