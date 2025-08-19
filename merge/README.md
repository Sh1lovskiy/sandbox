# merge

Modular RGB-D merge pipeline that reuses your `utils` package:
- RGB-D -> point cloud
- BASE transform and ROI crop
- FGR -> robust ICP -> safe Colored ICP
- Optional TSDF integration
- Final export and basic visualization
