import torch
from Modules.Utilities import colorText as txt
from Modules.Utilities import *
from Modules.DataHandler import *
from Modules.Engine import ViT
from timeit import default_timer as timer
from tqdm.auto import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

class Main():


    def __init__(self, data):
        self.parser = argparse.ArgumentParser(description="Vision transformers mini food classifier.")
        self.parser.add_argument("--epochs", type=int, help=f"Number of epochs.")
        self.parser.add_argument("--batch", type=int, help="Number of batches.")
        self.parser.add_argument("--trival", type=bool, help="Trival augmentation on train dataset.")
        self.parser.add_argument("--optimizer", type=str, help="Desire optimization algorithm. SGD/ADAM.")
        self.parser.add_argument("--lr", type=float, help="Value of learning rate in float.")
        self.parser.add_argument("--wd", type=float, default=0.3, help="Value of weight decay in float. (default=0.3)")
        self.parser.add_argument("--save", type=bool, default=False, help="Save the model after training process. (default=False)")
        self.parser.add_argument("--load", type=str, help="Model name to load.")
        self.parser.add_argument("--image", type=str, help="Image path to predict.")
        self.args = self.parser.parse_args()
        self.data = data
        self.writer = SummaryWriter()

    def trainNN(self, device, trainLoader, testLoader):
        vit = ViT().to(device)
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(params=vit.parameters(), lr= self.args.lr, weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(params=vit.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.wd)
        criterion = torch.nn.CrossEntropyLoss()
        modelStartTime = timer()
        for epoch in tqdm(range(self.args.epochs)):
            train_loss, train_acc = 0, 0
            for batch, (X, y) in enumerate(trainLoader):
                X, y = X.to(device), y.to(device)
                y_pred = vit(X)
                loss = criterion(y_pred, y)
                train_loss += loss
                train_acc += accuracyFunc(y_true=y, y_pred=y_pred.argmax(dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss /= len(trainLoader)
            train_acc /= len(trainLoader)
            print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
            vit.eval() 
            with torch.inference_mode():
                test_loss, test_acc = 0, 0
                for batch, (inputs, labels) in enumerate(testLoader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = vit(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss
                    test_acc += accuracyFunc(y_true=labels, y_pred=outputs.argmax(dim=1))
                test_loss /= len(testLoader)
                test_acc /= len(testLoader)
                print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
            self.writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss}, global_step=epoch)
            self.writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc}, global_step=epoch)
        self.writer.close()
        modelEndTime = timer()
        totalTime = modelEndTime - modelStartTime
        print(f"Train time on {device}: {totalTime:.3f} seconds")
        if self.args.save:
            print(txt("INFO -> Saving model.", "y"))
            saveModel(vit.state_dict())
    
    def loadNN(self, device, testLoader):
        print(txt("INFO -> Running on loaded model ...", "y"))
        modelStartTime = timer()
        dict = getModel(self.args.load)
        criterion = torch.nn.CrossEntropyLoss()
        if dict != None:
            vit = ViT().to(device)
            vit.load_state_dict(dict)
            vit.eval() 
            with torch.inference_mode():
                test_loss, test_acc = 0, 0
                for batch, (inputs, labels) in enumerate(testLoader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = vit(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss
                    test_acc += accuracyFunc(y_true=labels, y_pred=outputs.argmax(dim=1))
                test_loss /= len(testLoader)
                test_acc /= len(testLoader)
                print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
            modelEndTime = timer()
            totalTime = modelEndTime - modelStartTime
            print(f"Train time on {device}: {totalTime:.3f} seconds")


    def predict(self, device):
        labels = ["Pizza", "Stake", "Sushi"]
        print(txt("INFO -> Predicting ...", "y"))
        modelStartTime = timer()
        dict = getModel(self.args.load)
        input = data.prepairImage(self.args.image)
        if dict != None and input != None:
            #input.to(device)
            vit = ViT().to(device)
            vit.load_state_dict(dict)
            vit.eval() 
            with torch.inference_mode():
                output = vit(input.to(device))
                imageProbs = torch.softmax(output, dim=1)
                label = labels[torch.argmax(imageProbs, dim=1)]
                print(f"This image is a {label} with probability of {round(torch.max(imageProbs, dim=1).values.item() * 100, 2)}")


    def startNN(self):
        device = checkHardware()
        if self.args.image != None:
            if self.args.load != None:
                self.predict(device)
            else:
                print(colorText("Error -> Model should be passed to predict. check --help flag.", "r"))
        else:
            if argChecker(self.args):
                trainLoader, testLoader = self.data.generateLoaders(self.args.batch, self.args.trival)
                if self.args.load != None:
                     self.loadNN(device, testLoader)  
                else:
                    self.trainNN(device, trainLoader, testLoader)
            
            

if __name__ == "__main__":
    data = DataHandler()
    main = Main(data)
    main.startNN()
    
    