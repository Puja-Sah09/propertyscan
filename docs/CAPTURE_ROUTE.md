# Capture route

One walk, then one command. A fresh capture of the four-room plan below is meant to take under 15 minutes with the phone already in hand. The synthetic benchmark uses this same route.

## Device

Hold the phone in portrait, about 1.45 m off the floor, LiDAR running. Stay 0.7–0.95 m off each wall. Walk to within 0.25 m of every corner. A pass that stops short leaves a blind corner and the room does not close.

## LiDAR, Record3D or Stray Scanner

For each room, four wall passes, then one look up at the ceiling from the middle of the room:

1. South wall, camera aimed slightly down.
2. East wall.
3. North wall.
4. West wall.
5. Centre of the room, camera tilted up so the ceiling fills the upper half of the frame.

Then walk through every doorway and repeat. Come back to the room you started in so the loop can close. Do not walk a second storey into the same capture.

Stray Scanner exports a zip (`depth/`, `confidence/`, `odometry.csv`, `rgb.mp4`). Record3D exports a `.r3d`. Either one is the capture argument.

```
py -m propertyscan run CAPTURE.zip -o out/this-walk
```

## Photos

One US Letter sheet flat on the floor of each room, long edge parallel to a wall, printed side up. The black border and the off-centre mark are the scale and the facing. Nothing else in the room is used as a ruler.

From the far side of the sheet, aim at a point about halfway from the sheet to the wall. Take two frames per wall, the second shifted about a metre along the wall, so a door in the middle does not hide the base. Name them `wall-south.jpg`, `wall-south-b.jpg`, and the same for north, east, west.

Stand in the doorway and shoot into the next room. Name that frame `door-to-<room>.jpg`. That filename is the stitch. A door photo that does not see the sheet is dropped, and that link is missing from the plan.

```
py -m propertyscan run photos/ -o out/photos
```

`photos/capture.json` sets `"tier": "photos"`. Images live in `photos/rooms/<name>/`.

## Video

Film each room the way the stills are taken: stand on the far side of that room's letter sheet and keep the sheet in frame. The first frame of a room looks back through the door you came in by. The last frame looks through the door you leave by. Between rooms, look up for at least two frames so the sheet leaves the picture. That gap is how the next room starts. A room with two exits, here the hall, is entered again and the second visit ends on the other door. The bedroom's long walls are shot from 1.7 m back from the sheet. From the far wall those bases read about 18 cm long, which misses a 3 percent gate.

```
py -m propertyscan run video/ -o out/video
```

`video/capture.json` sets `"tier": "video"`. Frames are `video/frames/*.jpg` in order.

## magicplan

On the phone, open the same rooms in magicplan (free Starter). From the project, open Files and Sharing, then Statistics, and export CSV. Pass that file through. The command prints room areas from the file against the plan and skips a blank cell. It does not write a number the file does not contain.

```
py -m propertyscan compare out/plan.json magicplan.csv
```

The interior column is `area_without_walls` when the export has it, otherwise `area`.

## What this environment actually captured

The scored building is the ray caster walking the route above. The three Drive archives are real iPhone LiDAR in Stray Scanner layout, and they were run with the same command:

| Archive | Plan | Footprint | Ceiling |
|---|---|---|---|
| `single_room` | `reports/samples/single_room/` | 22.51 m² | not in the cloud |
| `single_scan_floor_only` | `reports/samples/single_scan_floor_only/` | 32.74 m² | not in the cloud |
| `single_scan_with_ceiling` | `reports/samples/single_scan_with_ceiling/` | 33.19 m² | 3.10 m |

They have no tape, so they are not a gate. This PC did not record another walk. magicplan was not installed, so there is no Statistics CSV.
