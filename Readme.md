This is the repo for the the group project for the course "Artificial Intelligence 2".

## Notes on the docker image
Use the **Dockerfile** to build an image with all the necessary dependencies.
The mimicgen library suggested using the following versions:
- python=3.8
- robosuite=1.4.1
- robomimic=0.3
- mujoco==2.3.2

## Links for the libraries we are using
- [Robosuite](https://robosuite.ai/) 
- [Robomimic](https://robomimic.github.io/docs/introduction/overview.html)
- [Mimicgen](https://mimicgen.github.io/)

## Dataset
- [Mimicgen](https://huggingface.co/datasets/amandlek/mimicgen_datasets)
- [Dataset_Structure](https://robomimic.github.io/docs/datasets/overview.html#dataset-structure)


## Notes on the xml wrapper
I have implemented an xml wrapper that converting MuJoCo 1.x format tags into MuJoCo 2.x format tags.
One of the classes in the file handles the the data loading based on the architecture of the YCB dataset and the other handles the conversion of the xml tags.
The wrapper is located in the `xml_wrapper` folder. 
The xml files for the custom objects are taken from the [YCB dataset](https://github.com/elpis-lab/YCB_Dataset/tree/main) and unloaded into the assets/ycb folder. 
