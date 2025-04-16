
# You can find DualDub's Demo Pages [here](https://anonymous.4open.science/w/DualDubDemo-DD18/).



# train-CASP 📎

Dualbench test set and a PyTorch Lightning solution to training CASP (The evaluation model).

CASP overview

<p align="center">
    <img src="images/CASP.PNG" alt="CASP Section Image">
</p>

## Usage 🚂

1. DualBench test set:
   The `video-to-soundtrack.jsonl` file is the test set for the video-to-soundtrack task. Each entry in the JSONL file contains the following keys:

   - `id`: A unique identifier for the entry.

   - `audiopath`: The file path to the corresponding audio file.

   - `speech`: The transcribed speech content, if available.

   - `videopath`: The file path to the corresponding video file.


​	This dataset is used to evaluate the performance of the model in generating soundtracks for videos.

​	samples:

```json

```



2. Training code: 

```bash
cd CASP
bash train.sh
```

3. Inference code: 

```bash
cd CASP
python inference_score_v2c_bench_gt.py
```





## TODO ✅

- [ ] inference code
  - [x] a simple inference code
  - [ ] a more frinedly inference code
- [x] training code
- [ ] 1500h checkpoint
- [ ] 3000h checkpoint
