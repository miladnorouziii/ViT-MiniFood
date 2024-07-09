# ViT MiniFood
This network is based on the basic model of the vision transformer and the patch size is 16. Due to the time-consuming nature and the need for a lot of resources, transfer learning has been used for this network. 

After the first run, the data and dataset folders will be automatically created and downloaded.
Use arguments to run the code. for example: 

``` Main.py --epochs 20 --batch 32 --trival True --optimizer Adam --lr 3e-3 --save True ```

Also use the ```--help ``` flag for more information.

You can access the accuracy and loss plots with tensor board generated in runs folder.
