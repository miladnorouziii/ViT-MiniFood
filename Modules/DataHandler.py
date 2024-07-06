import requests
import zipfile
from pathlib import Path
from .Utilities import colorText as txt
from tqdm.auto import tqdm
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image

class DataHandler():

    dataPath = Path("Data/")
    

    def makeDir(self):
        if self.dataPath.is_dir():
            print(txt("INFO -> Data directory exist.", "y"))
        else:
            print(txt("INFO -> Data directory does not exist. Generating directory ...", "y"))
            self.dataPath.mkdir(parents=True, exist_ok=True)
        

    
    def getData(self):
        self.makeDir()
        if not any(self.dataPath.iterdir()):
            print(txt("INFO -> Data directory is empty. Downloading data ...", "y"))
            with requests.get("https://github.com/miladnorouziii/Datasets/raw/main/Pizza-Stake-Sushi/pizza_steak_sushi.zip", stream=True) as response:
                totalLength = int(response.headers.get("content-length"))
                with tqdm(response.iter_content(chunk_size=2048), total=totalLength, unit="B", unit_scale=True) as bar:
                    with open(self.dataPath / "pizza_steak_sushi.zip", "wb") as file:
                        for data in response.iter_content(chunk_size=2048):
                            bar.update(len(data))
                            file.write(data)
            with zipfile.ZipFile(self.dataPath / "pizza_steak_sushi.zip", "r") as zip:
                zip.extractall(self.dataPath)
            print(txt("INFO -> Data downloaded successfully.", "g"))
            os.remove(self.dataPath / "pizza_steak_sushi.zip")
        else:
            print(txt("INFO -> Data found. Skipping downloading data ...", "y"))
    
    def generateLoaders(self, batch, trivalAugmentaion) -> DataLoader:
        self.getData()
        print(txt("INFO -> Generating dataloaders ...", "y"))
        trainDataTransformAug = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.TrivialAugmentWide(17),
            transforms.ToTensor() 
        ])
        testDataTransform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor() 
        ])
        if trivalAugmentaion:
            trainData = datasets.ImageFolder(root="Data/train", 
                                  transform=trainDataTransformAug, 
                                  target_transform=None) 
        else:
            trainData = datasets.ImageFolder(root="Data/train", 
                                  transform=testDataTransform, 
                                  target_transform=None)
        testData = datasets.ImageFolder(root="Data/test", 
                                 transform=testDataTransform)
        trainDataloader = DataLoader(dataset=trainData, 
                              batch_size=batch,
                              num_workers=os.cpu_count(),
                              shuffle=True)
        testDataloader = DataLoader(dataset=testData, 
                             batch_size=batch, 
                             num_workers=os.cpu_count(), 
                             shuffle=False)
        return trainDataloader, testDataloader
        
    def prepairImage(self, image):
        iamgePath = Path(image) 
        if iamgePath.is_file():
            print(txt(f"Error -> Image found, converting Image ...", "g"))
            dataTransform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ])
            img = Image.open(iamgePath)
            transformedImage = dataTransform(img)
            return transformedImage.unsqueeze(dim=0)
        else:
            print(txt(f"Error -> Image does not exist, check image name & it's extension type!", "r"))
            return None
            


