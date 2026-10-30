# Vela Prototype

Vela is a Python prototype of a partition-centric vector data management layer
for object-store-like storage. It supports local filesystems, Amazon S3, and
S3-compatible services such as MinIO.

The prototype includes immutable base/delta bundles, CAS-published manifests,
advisory routing, compaction and migration workflows, and exact or FAISS HNSW
indexes.

## Quick start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the demo and tests:

```bash
python3 -m examples.basic_demo
python3 -m unittest discover -s tests -v
```

## Main options

Use K-Means to place append-oriented record batches:

```python
from vela.partition.kmeans_policy import KMeansPartitionPolicy

policy = KMeansPartitionPolicy(n_clusters=8, random_state=0)
```

K-Means placement does not provide stable logical-ID routing for upserts or
tombstones.

Use the persistent FAISS HNSW index instead of the exact `FlatIndex` baseline:

```python
from vela.index import FaissHNSWIndex

index = FaissHNSWIndex(
    metric="l2",
    m=32,
    ef_construction=200,
    ef_search=64,
)
```

Vectors are stored as uncompressed row-major float32 `vectors.npy` objects.
Older manifests that reference `vectors.json` remain readable.

## S3-compatible storage

`S3ObjectStore` uses boto3's standard credential chain. The bucket must already
exist.

```python
from vela.storage.s3_object_store import S3ObjectStore

# MinIO or another S3-compatible service
store = S3ObjectStore(
    bucket="vela",
    prefix="development",
    endpoint_url="http://127.0.0.1:9000",
    addressing_style="path",
)

# Amazon S3: omit endpoint_url and addressing_style
store = S3ObjectStore(bucket="my-production-bucket", prefix="vela")
```

For the shared development MinIO instance, use the repository helper:

```bash
source scripts/minio_env.sh
scripts/minio_env.sh show
scripts/minio_env.sh init
```

The default namespace is `experiments/$USER`. Preview or delete objects only in
that namespace with:

```bash
scripts/minio_env.sh reset
scripts/minio_env.sh reset --yes
```

Run the optional S3 integration test with:

```bash
VELA_RUN_S3_INTEGRATION=1 python3 -m unittest tests.test_s3_object_store
```

## Benchmarks

The construction benchmarks compare incremental K-Means, batch-local semantic
partitioning, global FAISS clustering, and contiguous blocks. They support
local or S3 storage and default to FAISS HNSW indexes.

Run a SIFT smoke benchmark:

```bash
python3 -m benchmarks.sift_benchmark \
  --dataset-dir /home/yangguo/Dataset/sift \
  --database-limit 10000 \
  --partition-count 16 \
  --routing-top-k 1,4,8,all \
  --results-json benchmark-results/sift-10k.json
```

Use `python3 -m benchmarks.gist_benchmark` for GIST. Set `--database-limit 0`
to run a complete dataset; this requires substantially more memory, disk space,
and time.

Other benchmark tools:

- `benchmarks.sift_routing_benchmark`: compare routing centroids and fan-outs.
- `benchmarks.sift_evolution_benchmark`: compare global index rebuilds with
  partition migration strategies.
- `benchmarks.plot_query_breakdown`: plot query latency from benchmark JSON.

Common options include:

- `--index-backend flat|faiss-hnsw`
- `--storage-backend local|s3`
- `--storage-namespace NAME`
- `--no-compact` and `--keep-data`

Use `python3 -m <benchmark_module> --help` for the complete option list.
