# Rendering Modes

This document gives a compact overview of the main rendering modes supported by
MarineSTD-GS.

## 1. Dataset Render

This is the standard rendering mode after training. By default, it uses the
default evaluation split.

```bash
ns-marinestd-render dataset --load-config <CONFIG_YML> --rendered-output-names rgb --output-path <OUTPUT_DIR>
```

Typical outputs include:

- `rgb`
- `rgb_spatial_degraded`
- `rgb_spatiotemporal_degraded`
- `rgb_attenuation_map`
- `rgb_backscatter_map`
- `rgb_caustic_pattern`
- `depth`

You can also render the training split explicitly:

```bash
ns-marinestd-render dataset --load-config <CONFIG_YML> --split train --rendered-output-names rgb --output-path <OUTPUT_DIR>
```

Typical outputs are written under:

```text
<OUTPUT_DIR>/<split>/<rendered_output_name>/...
```

## 2. Interpolated Render

This mode renders an interpolated camera trajectory built from training or
evaluation cameras. By default, it writes a video. You can also switch to image
sequence output with `--output-format images`.

```bash
ns-marinestd-render interpolate --load-config <CONFIG_YML> --output-path <OUTPUT_VIDEO.mp4> --rendered-output-names rgb --disable-td True --disable-sd True
```

Typical outputs are:

- a rendered video by default, or an image sequence if `--output-format images`
  is used
- an interpolated Nerfstudio camera-path JSON `interpolated_camera_path.json`
  saved next to the training config `<CONFIG_YML>`

If `CONFIG_YML` is located at:

```text
.../marinestd-gs/<TIMESTAMP>/config.yml
```

then the exported Nerfstudio camera path is saved at:

```text
.../marinestd-gs/<TIMESTAMP>/interpolated_camera_path.json
```

For example, to render the interpolated trajectory as a video:

```bash
ns-marinestd-render interpolate --load-config <CONFIG_YML> --output-path <OUTPUT_VIDEO.mp4> --rendered-output-names rgb --disable-td True --disable-sd True
```

If `--output-format images` is used, then `--output-path` should be a directory-like
path. The rendered image sequence will be written under subfolders associated
with the requested `rendered-output-names`.

To render water-effect outputs along the same interpolated path, you must load
water parameters and disable the TD branch:

```bash
ns-marinestd-render interpolate --load-config <CONFIG_YML> --output-path <OUTPUT_VIDEO.mp4> --rendered-output-names rgb rgb_spatial_degraded --disable-td True --water-param-load-path <WATER_PARAM_PT>
```

In trajectory-based rendering modes, multiple `rendered-output-names` are
concatenated side by side into the same output video or image sequence.

## 3. Camera-Path Render

This mode renders a user-provided camera trajectory from a Nerfstudio
camera-path JSON. By default, it writes a video. You can also switch to image
sequence output with `--output-format images`.

```bash
ns-marinestd-render camera-path --load-config <CONFIG_YML> --camera-path-filename <CAMERA_PATH_JSON> --output-path <OUTPUT_VIDEO.mp4> --rendered-output-names rgb --disable-td True --disable-sd True
```

Here, `--camera-path-filename` should point to a Nerfstudio camera-path JSON,
for example the `interpolated_camera_path.json` exported by the `interpolate`
mode.

This mode is useful when you already have a saved camera path and want to
render that exact trajectory.

Typical output is a rendered video by default, or an image sequence if
`--output-format images` is used.

To render water-effect outputs along the same camera path, you must load water
parameters and disable the TD branch:

```bash
ns-marinestd-render camera-path --load-config <CONFIG_YML> --camera-path-filename <CAMERA_PATH_JSON> --output-path <OUTPUT_VIDEO.mp4> --rendered-output-names rgb rgb_spatial_degraded --disable-td True --water-param-load-path <WATER_PARAM_PT>
```

## 4. Spiral Render

This mode renders a spiral camera trajectory. By default, it writes a video.
You can also switch to image sequence output with `--output-format images`.

```bash
ns-marinestd-render spiral --load-config <CONFIG_YML> --output-path <OUTPUT_VIDEO.mp4> --rendered-output-names rgb --disable-td True --disable-sd True
```

Typical output is a rendered video by default, or an image sequence if
`--output-format images` is used.

To render water-effect outputs along the spiral path:

```bash
ns-marinestd-render spiral --load-config <CONFIG_YML> --output-path <OUTPUT_VIDEO.mp4> --rendered-output-names rgb rgb_spatial_degraded --disable-td True --water-param-load-path <WATER_PARAM_PT>
```

## 5. Water-Parameter-Based Advanced Rendering

MarineSTD-GS also supports:

- exporting per-frame water parameters
- re-rendering a scene with water effects using water parameters from the same scene
- transferring water effects across scenes using water parameters from another scene

These workflows are described in the `Advanced Applications` subsection of the
main [README](./README.md).
