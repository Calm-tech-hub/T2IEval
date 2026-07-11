# Generation Parameters (`-g` / `-G`)

`GenerationConfig` (defined in `src/t2i_eval/core/schema.py`) controls how prompts are turned into images. Global values come from `-g/--gen` or the `generation:` block inside config files. Per-evaluator overrides (`-G` or `evaluations.<name>.gen_args`) only replace the keys you specify; everything else falls back to the global config.

## Field reference

| Key | Type | Default | Description / tips |
| --- | --- | --- | --- |
| `prompt` | str | `""` | Base prompt text. Evaluators typically override this internally, so you seldom set it globally. |
| `negative_prompt` | str \| null | `null` | CLIP-style negative prompt; useful for removing artifacts. |
| `steps` | int | `50` | Number of diffusion steps. Larger values improve quality but cost time. |
| `guidance_scale` | float | `7.5` | CFG scale. Too high can cause over-saturation; 6–8 works for most SD models. |
| `seed` | int \| null | `null` | RNG seed for reproducibility. When null, generation uses random seeds per evaluator. |
| `width` / `height` | int \| null | `null` | Output resolution. Match model defaults (e.g., SDXL = 1024). |
| `num_images_per_prompt` | int | `1` | Number of images generated per prompt. Beware memory use when >4. |

## Usage patterns

### Global tuning

```bash
-g steps=30,guidance_scale=6.5,seed=123
```

### Per-evaluator overrides

```bash
-e geneval -G num_images_per_prompt=4
-e geneval -G steps=28,width=512,height=512
```

### Config-file equivalent

```yaml
generation:
  steps: 25
  seed: 42
evaluations:
  geneval:
    gen_args:
      num_images_per_prompt: 4
```

## Tips

- `steps` × `num_images_per_prompt` × number of prompts drives runtime. Reduce any of them to manage throughput.
- Always set `seed` when you need deterministic comparisons between models.
- Some evaluators provide their own generation template; overrides still apply but ensure resolutions match benchmark expectations.
