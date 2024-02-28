# VideoPeeker

## If you don't bother reading readme

Open Examples.py and check for useful examples.

## Useful imports (hopefully)

```
from ClassVideoInfo import VideoInformation
```
A very simple class to store basic information of a video. Aiming at maintaining consistency across function calls.

```
import BmsStatsScanner
```
BmsStatsScanner helps reading stats file from decoding VTM/ECM compressed videos.


```
import VideoDataset
```
Some codes that helps translating video classes and sequence names into the file paths in your database.

```
import GetBlock
import Interpolations
import Filters
```
These modules relate to fetching data from video files and performing basic filtering. **GetBlock** fetches blocks and template areas with integer coordinates. **Interpolations** fetches blocks with sub-pel coordinates. **Filters** offers functions to perform 2D linear filters under the context of a video file.

## Some example tasks
Simulations for cross-component prediction (**CCP**) and extrapolation filter-based intra prediction (**EIP**) are largely developed.

Matrix-based intra prediction (**MIP**) is yet to be developed at the time of writing.
