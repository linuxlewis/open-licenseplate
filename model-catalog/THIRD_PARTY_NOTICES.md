# Third-Party Model Notice

The model assets in this catalog are third-party works. They are distributed
under the GNU Affero General Public License, version 3 (AGPL-3.0), as stated
by the upstream model repository.

## Source

- Repository: `morsetechlab/yolov11-license-plate-detection`
- Pinned revision: `251a30d7daedca065f56e04b0af04052c907c68f`
- Source URL: `https://huggingface.co/morsetechlab/yolov11-license-plate-detection`
- License: AGPL-3.0
- License text: `model-catalog/licenses/AGPL-3.0.txt`

The catalog contains converted Core ML packages derived from the pinned source
weights. Each manifest records the exact source file URL, source checksum,
conversion tool versions, conversion arguments, inspected Core ML contract,
package tree checksum, archive checksum, and release asset name.

The upstream documentation reports possible train/test contamination and warns
that its metrics may be overestimated. Upstream metrics are not product performance claims.
Operators must validate a model on held-out data that represents their own
cameras and plates.

The application remains able to import a user-supplied model package and
manifest. The custom model import option remains required. Catalog download
support is an approved later extension, not a replacement for custom model
import.
