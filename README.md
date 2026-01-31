# ml

## pUp probability mapping

`model_meta.json` includes `classes`, `upClass`, and `upClassIndex`. Always read the ONNX output tensor named `probOutputName` (typically `probabilities`) and compute:

```
pUp = probabilities[upClassIndex]
```

Avoid using the `label` output for probabilities. If `probOutputName` is missing, fall back to the `probabilities` output name and still use `upClassIndex` for mapping.
