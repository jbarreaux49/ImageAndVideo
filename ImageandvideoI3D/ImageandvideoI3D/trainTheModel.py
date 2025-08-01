import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import random



#Class------------------------------------------------------------------------------------------------------------------------------------------------#

#inherit from pytorch
#Doc for DataSet: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class VideoDataset(Dataset):

    #This function will read the index file and their labels
    def __init__(self, index_file, transform=None, maxFrames=24):
        self.data=[]
        self.labels=[]
        self.transform=transform
        self.max_frames=maxFrames
        with open(index_file,"r") as f:
            for line in f:
                sequence_path, label = line.strip().split()
                self.data.append(sequence_path)
                self.labels.append(int(label))

    #Will return the number of video in the dataset
    def __len__(self):
        return len(self.data)


    #Basically , we will here take a random vidéo in the 4 variations of the idx video file in the index file 
    # For example idx=4 will give ASL20_frames/train\class_0\signer152_hello_0, and we will take à variation randomly in that 
    #and return a torch tensor
    def __getitem__(self, idx):
        video_path = self.data[idx]
        label = self.labels[idx]
        frames = []

        #To get the path of all of the variations
        augmentation_dirs = [
            os.path.join(video_path, aug) for aug in ["original", "flipped", "brightened", "contrasted"]
            if os.path.isdir(os.path.join(video_path, aug))
        ]

        #print(augmentation_dirs)

        if not augmentation_dirs:
            raise RuntimeError(f"No valid augmentation: {video_path}")

        #We will take on of them randomly, and take the frames
        selected_dir = random.choice(augmentation_dirs)
        frame_files = [
            f for f in sorted(os.listdir(selected_dir))
            if os.path.isfile(os.path.join(selected_dir, f)) and f.lower().endswith(('.jpg', '.png'))]


        #Be careful this can happen if you stop the data preparation process or if the extraction file contains an incomplete file (.part)
        if len(frame_files)==0:
            raise RuntimeError(f"No valid frames found in {selected_dir}")


        frame_files=frame_files[:self.max_frames] #just in case but not suppose to happen

        #To convert every frame into RGB, we will apply the transform in the main here
        for frame_file in frame_files:
            frame_path = os.path.join(selected_dir, frame_file)
            img=Image.open(frame_path).convert("RGB")
            if self.transform:
                img=self.transform(img)
            frames.append(img)

        while len(frames)<self.max_frames: #Same just in case
            frames.append(frames[-1])

        #T:Number of frames C:Number of channels and  H, W: Height,width 
        frames=torch.stack(frames, dim=0)
        frames=frames.permute(1,0,2,3)
        return frames,label



#Based on the doc: https://pytorch.org/ignite/generated/ignite.handlers.early_stopping.EarlyStopping.html
class EarlyStopping:
    """
    This class will be use to prevent overfitting 

    Args:
        patience: The number of consecutive epochs without improvement allowed before stopping
        delta: minimum change 

    """
    def __init__(self,patience=5,delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def check(self,valLoss):
        if valLoss<(self.best_loss-self.delta):
            self.best_loss=valLoss
            self.counter=0
        else:
            self.counter+=1
            if self.counter>=self.patience:
                self.stop=True



#Function------------------------------------------------------------------------------------------------------------------------------------------------#

def createAnIndex(dataDir, output,maxC=10):
    """
    We are creating here an index file listing the paths of directories containing sequences of frames

    Args:
        dataDir: Path to the directory containing class subfolders
        output: Path where the index file will be saved
        maxC: Maximum number of classes to include in the index file
    """
    with open(output,"w") as f:

        #Ti check at each subfolder of each class
        for idx,folder in enumerate(sorted(os.listdir(dataDir))):
            if idx>=maxC:  
                break

            #Just to check if everything is a file
            Thepath = os.path.join(dataDir,folder)
            if not os.path.isdir(Thepath):
                continue

            #We are looking at every sequence in every subfile
            for signer_folder in sorted(os.listdir(Thepath)):
                TheSignerPath = os.path.join(Thepath, signer_folder)
                if os.path.isdir(TheSignerPath):
                    f.write(f"{TheSignerPath} {idx}\n")
    print(f"Index file created!!")


#r3d_18 doc: https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html
def createModel(num):
    """
    We load the pre-imported I3D model (here it's r3D_18)

    Args:
        num: Number of class we want to use
    """
    model=r3d_18(weights='KINETICS400_V1')
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num)
    )
    print(f"Model initialized with {num} classes")
    return model


#Train model --------------------------------------------------------------------------------------------------------------------------#
def trainTheModel(model,train_loader,val_loader,epochs=20,lr=0.00005,save_path="trained_model.pth"):
    device = torch.device("cuda") #I have cuda but you have to verify if you have it to make it run 
    print(f"Using device: {device}")

    print("We are starting the training--------------------------------------------------------------")
    model=model.to(device)

    classCounts=Counter([label for _, label in train_loader.dataset])
    classWeights=[1.0/classCounts[i] for i in range(len(classCounts))] #We will count the number of sample for each class and create different weight for each of them
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(classWeights).to(device)) #Loss function 
    optimizer=optim.Adam(model.parameters(), lr=lr) #An optimizer that adjusts the learning rate
    earlyStopping=EarlyStopping(patience=5) #The class define at the beginning to stop if the model is not learning

    trainLosses=[]
    valLosses=[]
    valAccuracies=[]


    #Epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train() #Tranning mode
        runningLoss = 0.0

        #Batch
        for i,(inputs,labels) in enumerate(train_loader):
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad() #We have to do that between the different batch
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss+=loss.item()  #It's for the total loss 
            print(f"Batch {i+1}/{len(train_loader)}:Loss={loss.item():.4f}")

        avgTrainLoss = runningLoss / len(train_loader)
        trainLosses.append(avgTrainLoss)
        print(f"Epoch {epoch + 1}, Train Loss: {avgTrainLoss:.4f}")

        #Evaluation step with evaluation dataset 
        model.eval()
        val_loss=0.0
        correct=0
        total=0
        predictions=[]
        with torch.no_grad():
            for inputs,labels in val_loader:
                inputs,labels = inputs.to(device), labels.to(device)
                outputs=model(inputs)
                loss=criterion(outputs, labels)
                val_loss+=loss.item()
                _,predicted =torch.max(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.cpu().numpy())
        avgValLoss=val_loss/len(val_loader)
        val_accuracy=100*correct/total
        valLosses.append(avgValLoss)
        valAccuracies.append(val_accuracy)
        print(f"Validation Loss: {avgValLoss:.4f}, Accuracy: {val_accuracy:.2f}%")
        print(f"Prediction Distribution: {Counter(predictions)}")
        #End validation step 


        earlyStopping.check(avgValLoss)
        if earlyStopping.stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print("END---------------------------")
    torch.save(model.state_dict(),save_path)

    print(f"Model saved:{save_path}")

    #To see the final graph 
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label='Train Loss')
    plt.plot(valLosses, label='Validation Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.subplot(1, 2, 2)
    plt.plot(valAccuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.show()
#End train Model ---------------------------------------------------------------------------------------------------------------#



#Test-------------------------------------------------------------------------------------------------------------------------------#
def testTheModel(model, test_loader):
    """
    Tests the trained model on the test dataset 

    Args:
        model: The trained PyTorch model
        test_loader: DataLoader for the test dataset
    
    Returns:
        accuracy: Accuracy of the model
    """

    #You can test without cuda 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval() #Evaluation mode
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100*correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Prediction Distribution:{Counter(all_predictions)}")
    print(f"True Label Distribution:{Counter(all_labels)}")

#Test end ------------------------------------------------------------------------------------------------------------------------------------#



# Main --- #
if __name__ == "__main__":
    num_classes=10
    model_path = "trained_model_10_classes.pth"

    createAnIndex("ASL20_frames/train","train_index.txt",maxC=num_classes)
    createAnIndex("ASL20_frames/val","val_index.txt",maxC=num_classes)
    createAnIndex("ASL20_frames/val","test_index.txt",maxC=num_classes)


    #The resize is not really a necessity it s just in case i got a problem in data preparation
    #Normalize to align the value with the pretrained model (r3D_18)
    #Doc of transforms used: https://pytorch.org/vision/stable/transforms.html
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456,0.406], std=[0.229,0.224,0.225])
    ])


    # Train and Validation Datasets
    train_dataset = VideoDataset("train_index.txt",transform=transform,maxFrames=64)
    val_dataset = VideoDataset("val_index.txt",transform=transform,maxFrames=64)


    #Doc : https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)


    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found")
        model = createModel(num_classes)
        trainTheModel(model, train_loader, val_loader, epochs=20, lr=0.00005, save_path=model_path)
    else:
        print(f"Model file '{model_path}' found")

    test_dataset = VideoDataset("test_index.txt", transform=transform, maxFrames=64)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    model = createModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from '{model_path}'.")

    testTheModel(model, test_loader)

