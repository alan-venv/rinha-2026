# Variables

## Vector

- `VECTOR_DIMENSIONS`: vector size.
- `VECTOR_SCALE`: float-to-`i16` quantization scale.
- `VECTOR_LEN`: derived vector byte length.

## IVF/PQ Manipulaveis

- `IVF_CENTROIDS`: number of IVF centroids.
- `IVF_ITERATIONS`: centroid k-means iterations.
- `IVF_SAMPLE_MULTIPLIER`: IVF training sample multiplier.
- `PQ_LAYOUT`: `(subquantizers, dimensions_per_subquantizer)`.
- `PQ_ITERATIONS`: PQ codebook refinement iterations.
- `PQ_SAMPLE_MULTIPLIER`: PQ training sample multiplier.
- `PQ_CODEWORDS`: codewords per PQ subquantizer.

## IVF/PQ Nao Manipulaveis

- `IVF_MAGIC`: IVF/PQ binary format marker.
- `IVF_HEADER_LEN`: IVF/PQ header byte length.

## Runtime Search

- `NEAREST_COUNT`: number of nearest candidates used for score.
