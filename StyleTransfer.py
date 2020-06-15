import os #for path operations
import torch
import torchvision
from torchvision import models,transforms
from torch.autograd import Variable
import torch.nn as nn

from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

import copy # for deepcopy
import numpy as np

#Detect if cuda is available for GPU training otherwise it will use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

%matplotlib inline


transform =transforms.Compose([transforms.Resize([224,224]),#Resize images   
                                 transforms.ToTensor()])#Converts the input image to PyTorch tensor

#Load_image helper function
def loadimg(path=None): 
    img=Image.open(path)
    img=transform(img)
    img=img.unsqueeze(0)#Adding a Dimension to a Tensor
    return img

#Load images
content_img=loadimg("images/view.jpg")
# content_img=Variable(content_img).cuda()
print(content_img.shape)

style_img=loadimg("images/painting.jpg")
# style_img=Variable(style_img).cuda()
print(style_img.shape)


unloader = transforms.ToPILImage()  # Convert tensor to image

plt.ion()   #Turn on interactive mode to export multiple images

def imshow(tensor, title=None):  # Image display function
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # Removing the dimension we just added
    image = unloader(image)       # Convert to PLT
    plt.imshow(image)             # Display image
    if title is not None:      
        plt.title(title)
    plt.pause(0.001)              # pause a bit so that plots are updated


plt.figure()                                 # create figure
imshow(style_img, title='Style Image')       # display image

plt.figure()
imshow(content_img, title='Content Image')




####### Content loss ##########################################
class Content_loss(torch.nn.Module):
    def __init__(self,weight,target): #Constructor
        super(Content_loss,self).__init__()
        # we 'detach' the target content from the tree used 
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight=weight
        self.target=target.detach()*weight  # Separate the target from the calculation so that it doesn't have gradient
        self.loss_fn=torch.nn.MSELoss()
        
    def forward(self,input): 
        self.loss=self.loss_fn(input*self.weight,self.target) #Calculate loss using MSE
        return input
    
    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

####### gram matrix ###########################################
       # After convolution, feature map becomes [b, ch, h, w]. 
       # Convert it to [b, ch, h*w] and [b, h*w, ch]. Inner product -> [b, ch, ch]
class Gram_matrix(torch.nn.Module):
    def forward(self,input):
        a,b,c,d=input.size()
                            # a=batch size(=1)
                            # b=number of feature maps
                            # (c,d)=dimensions of a f. map (N=c*d)
        feature=input.view(a*b,c*d)# resise F_XL into \hat F_XL
        gram=torch.mm(feature,feature.t())
        # compute the gram product
        # we 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
        return gram.div(a*b*c*d)
        

####### Style loss ###########################################
class Style_loss(torch.nn.Module): 
    def __init__(self,weight,target):
        super(Style_loss,self).__init__()
        self.weight=weight
        self.target=target.detach()*weight
        self.loss_fn=torch.nn.MSELoss()
        self.gram=Gram_matrix() # get target_feature's gram_matrix
    
    def forward(self,input):
        self.Gram=self.gram(input.clone()) 
        self.Gram.mul_(self.weight)
        self.loss=self.loss_fn(self.Gram,self.target)
        return input
    
    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

####### Get the module ########################################
Use_gpu=torch.cuda.is_available()
cnn=models.vgg19(pretrained=True).features
if Use_gpu:
    cnn=cnn.cuda()

model=copy.deepcopy(cnn)

# desired depth layers to compute style/content losses :
content_layer=["Conv_3"]
style_layer=["Conv_1","Conv_2","Conv_3","Conv_4"]

content_losses=[]
style_losses=[]
conten_weight=1
style_weight=1000


new_model=torch.nn.Sequential()
model=copy.deepcopy(cnn)
gram=Gram_matrix()

if Use_gpu:
    new_model=new_model.cuda()
    gram=gram.cuda()
index=1 # increment every time we see a conv
for layer in list(model)[:8]: # loop through 8 layers
    if isinstance(layer,torch.nn.Conv2d):# if current layer is nn.Conv2d
        name="Conv_"+str(index)
        new_model.add_module(name,layer) # add layers to the model
                                        
        if name in content_layer: #if is content
            # add content loss:
            target=new_model(content_img).clone()# model(content_img) pass the content forward
            content_loss=Content_loss(conten_weight,target)# calculate content loss
            new_model.add_module("content_loss_"+str(index),content_loss)# add content loss to model
            content_losses.append(content_loss) # add content loss to the content_losses list
            
        if name in style_layer:# if is style
            # add style loss:
            target=new_model(style_img).clone() # pass the style forward
            target=gram(target)
            style_loss=Style_loss(style_weight,target) 
            new_model.add_module("style_loss_"+str(index),style_loss)
            style_losses.append(style_loss)
        
    if isinstance(layer,torch.nn.ReLU): #if current layer is nn.ReLU
            name="Relu_"+str(index) 
            new_model.add_module(name,layer)
            index=index+1
            
    if isinstance(layer,torch.nn.MaxPool2d):# if current layer is nn.MaxPool2d
            name="MaxPool_"+str(index)
            new_model.add_module(name,layer)


input_img=content_img.clone() #clone the content image as input
parameter=torch.nn.Parameter(input_img.data)
optimizer=torch.optim.LBFGS([parameter])

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')

epoch_n=50
epoch=[0] #counter 
while  epoch[0] <=epoch_n:
    def closure():
        optimizer.zero_grad() 
        style_score=0 
        content_score=0 
        parameter.data.clamp_(0,1) # Clamp if it exceed the range(0,1)
        new_model(parameter)
        for sl in style_losses: #loop over loss list, add losses up
            style_score+=sl.backward() 
        
        for cl in content_losses:  
            content_score+=cl.backward()
            
        epoch[0]+=1
        if epoch[0]%50==0:
            print("Eproch:{} Style Loss: {:4f} Conent Loss:{:4f}".format(epoch[0],style_score.item(),content_score.item()))
        return style_score+content_score

    optimizer.step(closure)

plt.figure()
imshow(input_img, title='Output Image')