# Technical report

Property scan to a dimensioned plan. LiDAR, a letter sheet in stills, or a letter sheet in video. One JSON contract, one SVG. No learned model.

## What is measured

Wall lengths, opening widths, floor area, and ceiling height, each with a 95 percent interval. Rooms are stitched by shared openings. On LiDAR, a crack, a hole, or a water stain becomes a scope line, and three named rules can flag concealed damage: moisture behind an intact finish, a stain within 0.45 m of a real ceiling, and a crack at least 0.4 m long with 8 mm of recess.

The world is Z-up. Record3D poses are OpenGL and are converted. Stray Scanner poses are already an OpenCV camera in a Y-up world; they are rotated so Z is up. Depth of 0 is invalid. Stray confidence below 2 is dropped.

## LiDAR

Consecutive frames are matched in the horizontal plane. Roll, pitch, and camera height stay on the pose that arrived. Loop closures are accepted only when the raw positions are already within 1.25 m, the yaw within 35 degrees, and the match agrees within 0.45 m and 12 degrees. An odometry prior keeps a bad loop from walking the phone across the building. If the solved path still moves by more than 0.45 m at the median, the update is thrown out and the captured poses are fused. On the synthetic walk that proposal was 2.41 m, so it was thrown out. On the three Stray Scanner archives it was 3.5 m, 3.8 m, and 7.1 m, and those were thrown out too.

Wall points are taken from a band 0.95 m to 1.60 m above the floor, and below the ceiling when a ceiling was actually seen. Rooms are the free space enclosed by those points. A gap is a door only when the walk crosses it. That is what keeps a mirror, which has no path through it, from becoming a door.

If the walls do not close, as in the Drive scans, the walked region inside the convex hull of the walls is kept and edges with no wall behind them are marked as openings. Those plans are footprints of an open scan, not a taped survey.

## The ceiling

The first estimator used the 5th and 95th percentiles. The ceiling is a thin mode, and the 95th percentile sits on the noise above it. On the synthetic hall that was 2.455 m against a tape of 2.440 m, an error of 1.52 cm, just outside the 1.5 cm gate. The replacement takes the outermost dense 1.5 cm bin and the median of the points within 2 cm of it. On the same 129 frames the hall is 2.442 m, an error of 0.15 cm. The worst room is 0.18 cm. A second walk shifted by 4 cm repeats the ceilings inside 1 cm. Both estimators are still in the code, so the before plan can be rebuilt.

A scan that never sees a ceiling, which is the floor-only Drive archive and the single-room archive, is marked `ceiling_observed: false`. The number in the plan is the visible wall, and the interval is widened to ±0.60 m. It is not reported as a ceiling.

## Photos and video

Photos have no pose. A US Letter sheet on the floor is the only scale. The long edge is parallel to a wall. Two frames per wall, plus a `door-to-<room>` frame for the stitch. The video tier does not carry one track across the building: every sheet is the same rectangle, and a dropped frame never gets its step back. Each visit is measured in that room's sheet frame. The sheet leaving the frame for two shots starts the next room. A later visit with the same wall lengths is the same room, and its last look is the doorway into the room that follows.

On the synthetic photo set the four rooms share one frame. The stitch follows the doorway the camera was aimed at, including a door the other room's frame did not pose. Footprint 54.43 m² against 53.01 m², overlap 0, walls inside 8 percent. The bedroom area is 16.67 m² against 15.96 m². A letter sheet a few metres from a wall cannot pin that wall tighter than a few centimetres, so the photo area interval is 5 percent of the area. The ceiling reads 2.40 m against 2.44 m and 2.47 m, inside the 15 cm photo interval. The tape is inside every reported interval.

On the synthetic video the four rooms are in one frame, stitched on those doorway looks. The stands match the photographs, except the bedroom's long walls, which are taken 1.7 m back from the sheet. From the far wall that base landed 18 cm long, outside the 3 percent gate. From 1.7 m it is 4.205 m against 4.200 m. The other walls are inside 3 percent. The worst is the living room's 5.084 m side against 5.000 m. Ceilings read 2.40 m against 2.44 m and 2.47 m, inside the 8 cm video interval. Areas are inside 6 percent. The tape is inside every reported interval. That plan is `reports/benchmark/video_plan/`.

## Intervals

LiDAR half-widths are 1.5 cm or 0.4 percent on a wall, 1.2 cm on an opening, and 1.8 cm on a ceiling. Photo walls are 4 cm or 8 percent, openings 6 cm, ceilings 15 cm, and the area interval is 5 percent of the area. Video walls are 2 cm or 3 percent, openings 6 cm, ceilings 8 cm, and the area interval is 6 percent. When the median grey of a frame is under 25, every interval is multiplied by 1.8. On the synthetic LiDAR plan, the photo plan, and the video plan the tape sits inside every reported interval.

## The Drive scans

These are real iPhone LiDAR exports, not the synthetic building. There is no tape, so they are not a gate.

The single-room walk has no ceiling. Two regions, footprint 22.51 m², including a 2.84 m side the scan never closed. The floor-only walk has no ceiling and six regions, footprint 32.74 m², with only 0.80–0.95 m of wall in view. Damage calls there (42 cracks, 16 holes, 12 stains) are on that short band. The ceiling walk has a dense plane 3.10 m above the floor and five regions, footprint 33.19 m². Two of those rooms top out near 2.3 m instead of 3.1 m.

## What fails on purpose

A framed mirror or a window with no return has no points beyond the glass and no path through it, so it is not opened. Full-height furniture can still be taken for a wall. A wet floor and a glass partition were not captured. Low light widens the interval; it does not invent points. A floor-only scan cannot report a ceiling.

## What was not done

The three iPhone LiDAR archives above are the field captures. This PC did not record a further walk. magicplan was not installed and no Statistics CSV was exported. `compare` exits 2 rather than invent a column. If a Statistics file is passed later, it compares room areas that are actually in the file. The interior column is `area_without_walls` when that export includes it. Blank cells are skipped.
