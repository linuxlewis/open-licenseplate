# Model Catalog Build

This directory contains the conversion and packaging tool for the approved
first-party model catalog. It is not an application runtime dependency.

The build downloads only these files from the pinned Hugging Face revision:

- `license-plate-finetune-v1n.pt`
- `license-plate-finetune-v1s.pt`
- `license-plate-finetune-v1m.pt`

The source revision is
`251a30d7daedca065f56e04b0af04052c907c68f`. The source license is
AGPL-3.0. The source file SHA-256 values are checked before conversion.

Use a temporary Python 3.12.14 environment. Do not add conversion packages to
the application dependencies:

```bash
BUILD_ROOT="$(mktemp -d /tmp/open-licenseplate-model-catalog.XXXXXX)"
uv venv --python 3.12.14 "$BUILD_ROOT/.venv"
uv pip sync \
  --python "$BUILD_ROOT/.venv/bin/python" \
  --python-version 3.12.14 \
  --python-platform aarch64-apple-darwin \
  --only-binary :all: \
  --require-hashes \
  tools/model_catalog/requirements-macos-arm64.lock
"$BUILD_ROOT/.venv/bin/python" tools/model_catalog/build.py \
  --output-dir "$BUILD_ROOT/output"
```

The lock file is generated from `requirements.in` with:

```bash
uv pip compile \
  --python-version 3.12.14 \
  --python-platform aarch64-apple-darwin \
  --only-binary :all: \
  --generate-hashes \
  --emit-index-url \
  --output-file tools/model_catalog/requirements-macos-arm64.lock \
  tools/model_catalog/requirements.in
```

The script exports Core ML `mlpackage` files with NMS enabled, inspects the
actual package input and output descriptions, removes volatile package
metadata, writes stable package identifiers, creates uncompressed
reproducible ZIP archives, and writes manifests, the catalog lock, and
checksums. It runs the application's package directory validator first and
uses that validated file set for both the package tree hash and archive.
The package tree checksum uses the same path-and-byte algorithm as the
application model importer.

The generated output is suitable for release upload. The committed
`model-catalog/` metadata must be updated from the generated `manifests/`,
`model-catalog-lock.json`, and `SHA256SUMS` files after a successful build.
