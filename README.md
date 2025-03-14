# Fine tuning to generate D3 code from images

## Data

The training data is generated using the `data/graphGenerator.ts` script. This
script generates a set of images and corresponding D3 code. The images and D3
code are stored in the `data/` folder.

### Creating training data

1. Install dependencies

```bash
npm install
```

2. Build the project

```bash
npm run build
```

3. Generate training data

```bash
npm run gen
```

Graphs will be generated in the `data/generated_graphs` folder. D3 code is
generated in the `data/generated_code` folder.

## Training

Training is done using the `main.py` script with model LoRa weights saved in the
`models/` folder.

```bash
python main.py
```

## Inference

Inference is done using the `inference.py` script with model LoRa weights saved
in the `models/` folder.

```bash
python inference.py --image path/to/chart.png --output generated_code.html
```

### Inference Options

- `--image`: Path to the input chart image (required)
- `--model`: Path to the fine-tuned model (default:
  "models/qwen2.5-vl-3b-d3js-finetuned-best")
- `--output`: Output file for the generated D3.js in html format (default:
  "generated_code.html")
- `--max_length`: Maximum length of generated code (default: 2048)
- `--use_original`: Use the original model without fine-tuned weights (flag, no
  value needed)

Example:

```bash
python inference.py --image data/generated_graphs/graph_1_bar.png --output my_d3_code.html
```

The script will load the fine-tuned model, process the input image, and generate
D3.js code that recreates the chart in the image. The generated code will be
saved wrapped in a `<script>` tag in a html file.
