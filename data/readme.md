### bbox_jitter.py

Custom augmentation method that randomly jitter and flip bounding boxes  
with given config parameters.  

Assumes file directory structure as drawn below.
```txt
- data
    - train
        - i.png
        ...
    - test
        - j.png
        ...
    train.json
```

Example usage can be written as below.

```bash
python3 bbox_jitter.py --json_file_name train.json --max_jitter_x 15.0 --max_jitter_y 15.0 \
    --random_flip true --augment_iters 1 --exception_list 6 21 35 41 241 254 256
```

#### Config variables

- json_file_name: name of the `train.json` file
- max_jitter_x: maximum jitter postion of bbox by x axis
- max_jitter_y: maximum jitter postion of bbox by y axis
- random_flip: if set true, vertically flip bbox with 50% prob.
- augment_iters: number of iterations for the augmentation loop
- exception_list: index of images those are to be excluded from augmentation.