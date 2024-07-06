from colorama import init
import os
import torch
from random import randint
from pathlib import Path


def colorText(text, color):
        init()
        colorCode = ""
        if color == "g":
            colorCode = "\033[32m"
        elif color == "r":
            colorCode = "\033[31m"
        elif color == "y":
            colorCode = "\033[33m"
        elif color == "c":
            colorCode = "\033[36m"
        elif color == "m":
            colorCode = "\033[35m"
        return f"{colorCode}{text}\033[0m"


def checkHardware():
    print(colorText("\nINFO -> Checking your hardware ...\n", "y"))
    device:torch.device = "cpu"
    try:
        os.system('nvidia-smi')
    except Exception as e:
        print(colorText(f"Error -> {e}\n", "r"))
    if torch.cuda.is_available():
        print(colorText("\nCUDA is available.", "g"))
        numberOfGpus = torch.cuda.device_count()
        print(colorText(f"Number of available GPUs: {numberOfGpus}", "g"))
        for i in range (numberOfGpus):
            gpuProperties = torch.cuda.get_device_properties(i)
            print(colorText(f"GPU{i}: {gpuProperties.name}, (CUDA cores: {gpuProperties.multi_processor_count})", "g"))
        device = torch.device("cuda")
    else:
        print(colorText("WARNING -> OOps! your GPU doesn't support required CUDA version. Running on CPU ...\n If you have dedicate GPU on your system check the link below to make sure you are using torch with cuda.\n https://pytorch.org/get-started/locally/ ", "r"))
        device = torch.device("cpu")
    return device

def argChecker(args) -> bool:
    state = True
    if args.epochs == None:
        print(colorText("Error -> Epochs should be pass to code. check --help flag.", "r"))
        state = False
    if args.batch == None:
        print(colorText("Error -> Batch size should be pass to code. check --help flag.", "r"))
        state = False
    if args.optimizer == None:
        print(colorText("Error -> Optimizer should be pass to code. check --help flag.", "r"))
        state = False
    if args.lr == None:
        print(colorText("Error -> Learning rate should be pass to code. check --help flag.", "r"))
        state = False
    return state

def accuracyFunc(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100 
        return acc
    
def saveModel(stateDict):
    modelPath = Path("Models/")
    if modelPath.is_dir():
        print(colorText("INFO -> Models directory exist.", "y"))
    else:
        print(colorText("INFO -> Models directory does not exist. Generating directory ...", "y"))
        modelPath.mkdir(parents=True, exist_ok=True)
    randNum = randint(10000, 99999)
    filename = f'MN-FoodRecognition-{str(randNum)}.pth'
    torch.save(stateDict, modelPath / filename)
    print(filename)

def getModel(fileName):
    modelPath = Path("Models/"+ fileName + ".pth") 
    if modelPath.is_file():
        return torch.load(modelPath)
    else:
        print(colorText(f"Error -> Model does not exist, check model name!", "r"))
        return None


