# Posturiser

- All old code from 2023-2024 is in old_code/
- Feb (or onwards) submission's work will be done in various root dir folders.


#### Project Direction
- Feel that the strategies laid out in the original EPQ needs to be re-evaluated.
- New strategies we can take up:
  - Use some type of baseline (mediapipe, or mediapipe face mesh) -> use that to compute metrics like pitch etc.
  - Feature extractor -> classification head
- Fix current strategy with:
  - Replacing things like “30 px” with T * face_height or delta/face_height.
  - Adding persistence in some way, for example alert only if bad posture for k consecutive seconds/frames.

  Then we can compute and compare metrics like accuracy, precision and recall, runtime (fps/cpu usage?), and how robust it is in terms of handling changes to lighting, distance or the camera angle.