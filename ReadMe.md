# METNet: A mesh exploring approach for segmenting 3D textured urban scenes

## Description

This is a implementation of the paper: [METNet] (https://doi.org/10.1016/j.isprsjprs.2024.10.020)

## Installation
This project was tested on Ubuntu 24.04.1 with CUDA 11.4 on a GeForce RTX 2080 Ti.

### Requirements:
- Python 3.7
- PyTorch 1.10.0
- h5py 3.8.0
- scikit-learn 1.0.2
- NumPy, Pillow, plyfile

**Recommended:**  
Use a virtual environment to manage dependencies.

### Usage

## Data Organization

The SUM data can be downloaded from [SUM](https://3d.bk.tudelft.nl/projects/meshannotation/). Copy the training, validation, and test folders into the raw_data folder.

## MET Generation 

```bash
python met_generation.py --mode train --data_usage 0.1 # Percentage of training data to be used. 0.1 corresponds to 10%.
python met_generation.py --mode val
python met_generation.py --mode test
```

## Training

```bash
python train.py --epochs 30 --batch_size 4096 --model_name model.pt --num_workers 16
```

## Test

```bash
python test.py --model_name model.pt
```

## Citation

If you use this in your research or work, please cite it:

```
@article{metnet2024,
	author = {Qendrim Schreiber and Nicola Wolpert and Elmar Schömer},
	title = {METNet: A mesh exploring approach for segmenting 3D textured urban scenes},
	journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
	volume = {218},
	pages = {498-509},
	year = {2024},
	issn = {0924-2716},
	doi = {https://doi.org/10.1016/j.isprsjprs.2024.10.020},
	url = {https://www.sciencedirect.com/science/article/pii/S0924271624003976}
}
```

## License

This project is licensed under the MIT License.

