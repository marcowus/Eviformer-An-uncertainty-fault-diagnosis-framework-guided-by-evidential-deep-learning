# Eviformer
# Eviformer: An uncertainty fault diagnosis framework guided by evidential deep learning
* Code about evidential deep learning and uncertainty quantification
* Core codes for the paper:
<br> [Eviformer: An uncertainty fault diagnosis framework guided by evidential deep learning](https://www.sciencedirect.com/science/article/abs/pii/S095219762502336X?dgcid=coauthor)
* Created by Jingjie Luo , Fucai Li *, Xiaolei Xu.
* Journal: Engineering Applications of Artificial Intelligence
* Schematic diagram of the Distribution Measurer
<div align="center">
<img src="https://github.com/Luojunqing666/Eviformer-An-uncertainty-fault-diagnosis-framework-guided-by-evidential-deep-learning/blob/main/Schematic%20diagram%20of%20the%20Distribution%20Measurer.png" width="600" />
</div>  
* The overall procedure of the proposed method
<div align="center">
<img src="https://github.com/Luojunqing666/Eviformer-An-uncertainty-fault-diagnosis-framework-guided-by-evidential-deep-learning/blob/main/The%20overall%20procedure%20of%20the%20proposed%20method.png" width="600" />
</div>  

## Our operating environment
* Python 3.8
* pytorch  1.10.1
* numpy  1.22.0 (If you get an error when saving data, try lowering your numpy version!)
* and other necessary libs

## Datasets
* Any publicly available dataset that includes bearing or gear vibration signals can use this method.

## Refactored CLI (v1)

The repository now contains a standalone, script-style interface that does not rely on
Python packages or `__init__.py` files. The new entry point is
`Eviformer/cli_v1.py`, which can be executed directly:

```
python Eviformer/cli_v1.py train \
  --data-root "D:/CWRU_Bearing_NumPy-main/Data" \
  --model mcswint \
  --epochs 100
```

Key features of the refactored pipeline include:

* **Scan mode** for reproducing the verbose directory listing shown in the data
  description:

  ```
  python Eviformer/cli_v1.py scan --data-root "D:/CWRU_Bearing_NumPy-main/Data"
  ```

* **Inspect mode** for printing dataset metadata without launching training:

  ```
  python Eviformer/cli_v1.py inspect --data-root "D:/CWRU_Bearing_NumPy-main/Data"
  ```

* **Train/Test modes** that share a simplified configuration surface while
  maintaining the original evidential learning capabilities. Checkpoints are
  stored at `./results/model_v1.pt` by default.

All helper modules that support the CLI follow the `*_v1.py` naming convention
for clarity (`data_pipeline_v1.py`, `model_factory_v1.py`, `training_loop_v1.py`,
`evaluation_v1.py`, and `sequence_dataset_v1.py`).

* ## Citation
If our work is useful to you, please cite the following paper, it is the greatest encouragement to our open source work, thank you very much!
```
@paper{
  title = {Eviformer: An uncertainty fault diagnosis framework guided by evidential deep learning},
  author = {Jingjie Luo, Fucai Li, Xiaolei Xu, Wenqiang Zhao, Dongqing Zhang},
  journal = {Engineering Applications of Artificial Intelligence},
  volume = {161},
  pages = {112328},
  year = {2025},
  doi = {doi.org/10.1016/j.engappai.2025.112328},
  url = {https://www.sciencedirect.com/science/article/abs/pii/S095219762502336X},
}
```

## Contact
- luojingjie@hnu.edu.cn
- luojingjie@sjtu.edu.cn
